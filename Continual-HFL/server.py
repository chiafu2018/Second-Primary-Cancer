import os
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Global settings
rounds = 10
min_client = 2
total_feature_number = 27

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The server first request initialized parameters from one of the clients
# And then send the initialized parameters to the rest of the clients 


# Define PyTorch Model
class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.LayerNorm(12)
        self.fc3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(12, 6)
        self.fc5 = nn.Linear(6, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.hardshrink(self.fc1(x)) 
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x



# Function to get model parameters
def get_model_parameters(model: nn.Module):
    """Extracts model parameters as a list of NumPy arrays."""
    return [param.detach().cpu().numpy() for param in model.state_dict().values()]


def fit_config(rnd: int):
    """Defines the configuration for training in each round."""
    config = {
        "round": rnd,
        "local_epochs": 100 if rnd <= 5 else 200
    }
    
    return config


# Main Function to Start Flower Server
def main() -> None:
    model = Net(input_size=total_feature_number, output_size=2)
    model.to(DEVICE)  # Ensure model is on the correct device

    strategy = fl.server.strategy.FedAdam(
        min_fit_clients=min_client,
        min_evaluate_clients=min_client,
        min_available_clients=min_client,
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(model))
    )

    fl.server.start_server(
        server_address="127.0.0.1:6000",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy
    )


if __name__ == "__main__":
    main()
