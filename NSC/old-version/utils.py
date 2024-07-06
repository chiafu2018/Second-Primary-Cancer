'''
This code serves as a backup in case fine-tuning fails. 
'''

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam


class SpcancerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train.astype(float), y_train.astype(float)
        self.x_test, self.y_test = x_test.astype(float), y_test.astype(float)

        
    def get_parameters(self):
        """Get parameters of the local model."""
        return self.model.get_weights()
        # raise Exception("Not implemented (server-side parameter initialization)")


    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters
        self.model.set_weights(parameters)

        # Train the model using hyperparameters from config
        class_weights = {0: 1, 1: 5} 
        history = self.model.fit(self.x_train, self.y_train, epochs = 40, class_weight=class_weights)
        
        # Return updated model parameters and results
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
        }

        print(f"Round: {config['rnd']}")
        return self.model.get_weights(), len(self.x_train), results


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model with global parameters
        self.model.set_weights(parameters)
        # Use aggregate model to predict test data
        y_pred = self.model.predict(self.x_test)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        results = {"loss" : loss, "accuracy": accuracy, "auc": roc_auc_score(self.y_test, y_pred)}

        return loss, len(self.x_test), results
