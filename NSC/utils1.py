
'''
You can modify the following algorithm (code) as you want. 
The old-version folder has the back-up. 
'''

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from sklearn.metrics import roc_auc_score, fbeta_score, roc_curve
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class SpcancerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train.astype(float), y_train.astype(float)
        self.x_test, self.y_test = x_test.astype(float), y_test.astype(float)

    def get_parameters(self):
        """Get parameters of the local model."""
        return self.model.get_weights()


    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters
        self.model.set_weights(parameters)

        # Train the model using hyperparameters from config
        class_weights = {0: 1, 1: 3}
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 40)

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping]
        )
        
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
        y_pred_prob = self.model.predict(self.x_test)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Evaluate global model parameters on the local test data
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        auc = roc_auc_score(self.y_test, y_pred_prob)
        f1 = fbeta_score(self.y_test, y_pred, beta=2)

        results = {
            "loss": loss,
            "accuracy": accuracy,
            "auc": auc,
            "f1_score": f1
        }

        return loss, len(self.x_test), results

