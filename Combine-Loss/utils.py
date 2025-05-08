import pandas as pd
import argparse
import numpy as np
import flwr as fl
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from collections import OrderedDict
from train import federateTrain
from sklearn.metrics import roc_auc_score
import shap
from typing import List, Tuple
import torch 

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(DEVICE)


def set_parameters(net, parameters: List[np.ndarray]):
    """
    Updates the model's parameters (state_dic) using a provided list of NumPy arrays.

    Functionality:
    - Extracts the model's parameter keys from `state_dict()` and pairs them with the new provided `parameters`.
    - Updates the model's parameters by loading the newly created `state_dict`.
    - Uses `strict=False` to allow partial updates, meaning that layers not present in the `parameters` list 
      will retain their original values. By doing so, we can prevent the error. 
    """
    
    # Map model's parameter keys to provided parameter values
    params_dict = zip(net.state_dict().keys(), parameters)

    # Create an ordered dictionary with new parameter values as PyTorch tensors
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    # Load the new state dictionary into the model (allowing partial updates if necessary)
    net.load_state_dict(state_dict, strict=False)


def get_parameters(net) -> List[np.ndarray]:
    """
    Extracts the model's parameters and returns them as a list of NumPy arrays.

    Functionality:
    - Calls `state_dict()` to retrieve the model's parameters as an ordered dictionary.
    - Converts each parameter tensor to CPU (if on GPU) to avoid device conflicts.
    """

    # Convert all parameters to CPU first (if on GPU) and then to NumPy arrays
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class SpcancerClient(fl.client.NumPyClient):
    def __init__(self, net, x_train, y_train, class_weights):
        self.net = net
        self.x_train = x_train
        self.y_train = y_train
        self.class_weights = class_weights

    def get_parameters(self):
        return get_parameters(self.net)


    def fit(self, parameters, config):
        """
        Updates the model's parameters, trains it locally, and returns updated parameters.

        Args:
            parameters: Parameters sent by the server to update to local net
            config: Information sent by the server in every round

        Functionality:
        - Sets the model's parameters using those received from the server.
        - Trains the model locally on the client's dataset.
        - Returns the updated model parameters, training sample count, and any additional metrics.
        """        
        
        # Update parameters received from server after aggregation
        set_parameters(self.net, parameters)

        # Train locally using the specified number of local epochs and class weights
        loss, auroc = federateTrain(self.net, self.x_train, self.y_train, epochs=config["local_epochs"],  class_weights=self.class_weights)

        return get_parameters(self.net), len(self.x_train), {"loss": loss, "auroc": auroc}


def targetEncode(series, target, alpha=3):
    series, target = pd.Series(series), pd.Series(target)
    
    global_mean = target.mean()
    
    df = pd.DataFrame({'feature': series, 'target': target})
    agg = df.groupby('feature')['target'].agg(['count', 'mean'])
    smooth = ((agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)) * 10


    default_value = (alpha * global_mean / (alpha)) * 10

    min_val = min(smooth.min(), default_value)
    max_val = max(smooth.max(), default_value)

    scaled = (smooth - min_val) / (max_val - min_val) 
    default_value = (default_value - min_val)/(max_val - min_val) 


    encoded = series.map(scaled).fillna(default_value)

    return encoded, {
        "mapping": scaled.to_dict(),
        "default": default_value,
        "feature_name": series.name
    }


def scaling(series):
    min_val = series.min()
    max_val = series.max()

    scaled = (series - min_val) / (max_val - min_val) 
    return scaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def get_class_balanced_weights(y_train, beta):
    # Count the number of samples for each class
    class_counts = Counter(y_train)

    # Calculate the effective number for each class
    effective_num = {}
    for class_label, count in class_counts.items():
        effective_num[class_label] = (1 - beta**count) / (1 - beta)

    # Calculate the class-balanced weight 
    scaling = 10000
    class_weights = [(1 / effective_num[0]) * scaling, (1 / effective_num[1]) * scaling]


    print(f"SPC False: {class_weights[0]}   SPC True: {class_weights[1]}")
    return torch.FloatTensor(class_weights).cuda().to(DEVICE)




def parse_argument_for_running_script():
    parser = argparse.ArgumentParser(description="Training Script for a Federated Learning Model")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--hospital', type=int, default=1, help='Hospital Data for training')
    args = parser.parse_args()
    return args.hospital, args.seed



def featureInterpreter(name, model, x_train, institution, method, seed):
    hospital = 'Taiwan' if institution == 1 else 'USA'

    background_data = shap.sample(x_train, 100)
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(x_train.iloc[299:399, :])

    # shap summary plot 
    # shap.summary_plot(shap_values, x_train.iloc[299:399, :], show=False)
    
    # shap summary beeswarm plot (yes class)
    shap.summary_plot(shap_values[1], x_train.iloc[299:399, :], show=False)

    plt.subplots_adjust(top=0.85) 
    plt.title(f'{name} | {hospital} | summary | seed = {seed}')
    plt.savefig(f'Results/shap/{name}_{hospital}_{seed}.png')
    plt.close('all')
