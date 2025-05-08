import numpy as np
import pandas as pd
import optuna 
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import utils

EPOCHS = 1500

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"PyTorch {torch.__version__}")


def dataTypePreprocess(dataframe):
    x = torch.from_numpy(dataframe.drop(columns=['Target']).values).float().to(DEVICE)
    y = torch.from_numpy(utils.to_categorical(dataframe['Target'], num_classes=2)).float().to(DEVICE)
    return x, y


class Net(nn.Module):
    def __init__(self, input_size) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.softmax(self.fc1(x), dim=1)
        return x


def objective(trial, x_train, y_train, class_weights):
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-4)
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-4)
    
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []

    for train_idx, val_idx in skf.split(x_train.detach().cpu().numpy(), y_train[:,1].detach().cpu().numpy()):

        x_tr, x_val = x_train[train_idx], x_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = Net(input_size=x_train.shape[1]).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-20)

        model.train()
        for epoch in range(EPOCHS):  
            optimizer.zero_grad()
            outputs = model.forward(x_tr)
            loss = criterion(outputs, y_tr)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item()) 

        model.eval()
        with torch.no_grad():
            outputs = model(x_val)

        aurocs.append(roc_auc_score(y_val[:, 1].cpu().numpy(), outputs[:, 1].cpu().numpy()))

    return np.mean(aurocs)


def train(net, x_train, y_train, class_weights, best_params, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
    optimizer = torch.optim.AdamW(net.parameters(), weight_decay=best_params['weight_decay'], lr=best_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-20)

    net.train()
    for epoch in range(EPOCHS):

        optimizer.zero_grad()    # 1. Zero out old gradients 
        outputs = net.forward(x_train)  # 2. Foward pass 
        loss = criterion(outputs.float().to(DEVICE), y_train.float())   # 3. Compute loss
        loss.backward()     # 4. Backward pass (compute gradients)
        optimizer.step()    # 5. Update weights
        scheduler.step(loss.item())    # 6. Update learning rate
        
        # Metrics
        if verbose:
            print(f"Epoch: {epoch+1} | Loss {loss.item():.4f}")

    for param_group in optimizer.param_groups:
        print("Learning Rate:", param_group['lr'])


def evaluateModel(net, x_test, y_test):
    net.eval()
    with torch.no_grad():
        outputs = net.forward(x_test)
    
    probability = outputs[:, 1].cpu().numpy()
    y_test = y_test[:, 1].cpu().numpy()

    if (sum(y_test)):
        auprc = average_precision_score(y_true=y_test, y_score=probability)
        auroc = roc_auc_score(y_true=y_test, y_score=probability)
    else:
        auprc, auroc = None, None

    return {
        'auprc': auprc, 
        'auroc': auroc
    }


def main():
    '''
    If you use the script to run this program, where you can test multiple seeds per time. You need to comment 
    LINE: institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))
    Otherwise, you need to comment, where you can only test for one seed.
    LINE: institution, seed = utils.parse_argument_for_running_script()
    '''
    institution, seed = utils.parse_argument_for_running_script()
    # institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 22
    
    df_train = pd.read_csv(f'Middle/Train_{institution}.csv')
    trainset = utils.csvPreprocess(df_train)

    df_test = pd.read_csv(f'Middle/Test_{institution}.csv')
    testset = utils.csvPreprocess(df_test)


    # class weights
    beta = (len(trainset['Target'])-1)/len(trainset['Target'])
    class_weights = utils.get_class_balanced_weights(trainset['Target'], beta)


    x_train, y_train = dataTypePreprocess(trainset)
    x_test, y_test = dataTypePreprocess(testset)


    # Hyperparameter Optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
    opt_objective = partial(objective, x_train=x_train, y_train=y_train, class_weights=class_weights)
    study.optimize(opt_objective, n_trials=100)
    best_params = study.best_params 

    # Retrain model with best hyperparameter 
    model = Net(input_size=x_train.shape[1]).to(DEVICE)
    train(model, x_train=x_train, y_train=y_train, class_weights=class_weights, best_params=best_params)

    metrics_results = evaluateModel(model, x_test, y_test)

    results = pd.DataFrame([{
        'Seed': seed,
        'AUROC': metrics_results['auroc'],
        'AUPRC': metrics_results['auprc']
    }])

    hospital = 'Taiwan' if institution == 1 else 'USA'

    results.to_csv(f'Results/NN_{hospital}.csv', mode='a', index=False)
    print("results saved to NN.csv")

if __name__ == "__main__":
    main()
