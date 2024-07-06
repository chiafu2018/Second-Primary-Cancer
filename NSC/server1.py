'''
You can modify the following algorithm (code) as you want. 
The old-version folder has the back-up. 
'''

from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Global settings
lr_rate = 0.03
min_client = 2
rounds = 5
total_feature_number = 43

def main() -> None:
    
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(total_feature_number, ), name='base1'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='base2'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid', name='personal'))
  
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients = min_client,
        min_eval_clients = min_client,
        min_available_clients = min_client,
        on_fit_config_fn = fit_config,
        on_evaluate_config_fn = evaluate_config,
        initial_parameters = fl.common.weights_to_parameters(model.get_weights())
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("127.0.0.1:6000", config={"num_rounds": rounds}, strategy=strategy)
    

def fit_config(rnd: int):
    config = {
        "rnd": rnd,
        "batch_size": 16,
        "local_epochs": 50
    }
    return config

def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps, "rnd": rnd}


if __name__ == "__main__":
    main()
