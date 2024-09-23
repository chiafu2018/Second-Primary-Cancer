from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Global settings
lr_rate = 0.03
min_client = 2
rounds = 5
total_feature_number = 43

def main() -> None:
    
    model = Sequential() 
    model.add(Dense(12, activation = 'relu', input_shape=(total_feature_number,)))
    model.add(BatchNormalization())
    model.add(Dense(6, activation = 'relu'))  
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=['accuracy'])

    strategy = fl.server.strategy.FedAdam(
        min_fit_clients = min_client,
        min_eval_clients = min_client,
        min_available_clients = min_client,
        on_fit_config_fn = fit_config,
        on_evaluate_config_fn = evaluate_config, 
        initial_parameters = fl.common.weights_to_parameters(model.get_weights())
    )

    fl.server.start_server("127.0.0.1:6000", config={"num_rounds": rounds}, strategy=strategy)
    

def fit_config(rounds: int):
    config = {
        "round": rounds,
        "local_epochs": 40 if rounds < 3 else 60
    }
    return config

def evaluate_config(rounds: int):
    config = {
        "val_steps": 5 if rounds < 3 else 10
    }
    return config


if __name__ == "__main__":
    main()
