'''
This code serves as a backup in case fine-tuning fails. 
'''

from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

lr_rate = 0.3
min_client = 2
rounds = 5
total_feature_number = 46


def main() -> None:
    
    model = Sequential() 
    model.add(Dense(12, activation='relu', input_shape=(total_feature_number, ), name='base1'))
    model.add(Dense(6, activation='relu', name='base2'))
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
    fl.server.start_server("127.0.0.1:6000", config={"num_rounds":rounds}, strategy=strategy)
    

def fit_config(rnd: int):
    config = {
        "rnd": rnd,
        "batch_size": 16,
        "local_epochs": 50
    }
    return config

def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps, "rnd" : rnd}


if __name__ == "__main__":
    main()


