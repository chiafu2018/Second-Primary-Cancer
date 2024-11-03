import pandas as pd
import argparse
import flwr as fl
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import shap


class SpcancerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, class_weights):
        self.model = model
        self.x_train, self.y_train = x_train.astype(float), y_train.astype(float)
        self.x_test, self.y_test = x_test.astype(float), y_test.astype(float)
        self.class_weights = class_weights

    def get_parameters(self):
        return self.model.get_weights()


    # config is the information which is sent by the server every round.
    # The content of the config will change every round
    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        print(f"Round: {config['round']}")
        epochs: int = config["local_epochs"]

        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000005)
        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, class_weight=self.class_weights, callbacks=[lr_scheduler])

        # draw_loss_function(history=history, name="federated learning")

        # Return updated model parameters and results
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
        }

        return self.model.get_weights(), len(self.x_train), results


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        pred_prob = self.model.predict(self.x_test)

        loss, accuracy = self.model.evaluate(self.x_test, to_categorical(self.y_test, num_classes=2), steps = config['val_steps'])
        auc = roc_auc_score(self.y_test, pred_prob[:, 1])

        results = {
            "accuracy": accuracy,
            "auc": auc,
        }

        return loss, len(self.x_test), results



def get_class_balanced_weights(y_train, beta):
    # Count the number of samples for each class
    class_counts = Counter(y_train)

    # Calculate the effective number for each class
    effective_num = {}
    for class_label, count in class_counts.items():
        effective_num[class_label] = (1 - beta**count) / (1 - beta)

    # Calculate the class-balanced weight
    class_weights = {class_label: (1 / effective_num[class_label]) for class_label in class_counts}
    return class_weights


def draw_loss_function(history, name):
    try:
        plt.plot(history.history['loss'])
    except:
        plt.plot(history[0], history[1]) #adding this for seesawing weights 

    plt.title(f'Model loss -- {name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc = 'upper left')
    plt.show()


def parse_argument_for_running_script():
    parser = argparse.ArgumentParser(description="Training Script for a Federated Learning Model")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--hospital', type=int, default=1, help='Hospital Data for training')
    args = parser.parse_args()
    return args.seed, args.hospital


def featureInterpreter(name, model, x_train, institution, method, seed):
    hospital = 'Taiwan' if institution == 1 else 'USA'

    background_data = shap.sample(x_train, 100)
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(x_train.iloc[299:399, :])

    shap.summary_plot(shap_values, x_train.iloc[299:399, :], show=False)

    plt.subplots_adjust(top=0.85) 
    plt.title(f'{name} | {hospital} | summary | seed = {seed}')
    plt.savefig(f'Results/shap/{name}_{hospital}_{seed}.png')
    plt.close('all')



def featureInterpreter_SSW(ser_weight, loc_weight, institution, seed):
    hospital = 'Taiwan' if institution == 1 else 'USA'

    feature_result = pd.DataFrame({
        'feature': ['global model predict no prob', 'global model predict yes prob', 
                    'local model predict no prob', 'local model predict yes prob'],
        'weight': [ser_weight, ser_weight, loc_weight, loc_weight]
    })
    
    plt.subplots_adjust(left=0.35)
    sns.barplot(x='weight', y='feature', data=feature_result)
    plt.xlabel("Weight")

    plt.title(f'SSW | {hospital} | summary | seed = {seed}')
    plt.savefig(f'Results/shap/SSW_{hospital}_{seed}.png')
    plt.close('all')