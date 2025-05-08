import optuna
import flwr as fl
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import utils

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")


'''''''''''''''''''''''''''''''''' Feature Groups '''''''''''''''''''''''''''''''''''''''

global_feature = ['Laterality', 'Age', 'Gender', 'SepNodule', 'PleuInva', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 
                 'Chemotherapy', 'Surgery', 'PTHLTYPE_Histology']

# Ensure different clients with same features after one hot
one_hot_feature = ['PleuInva_1', 'PleuInva_2', 'PleuInva_9', 'Gender_1', 'Gender_2',
                     'SepNodule_1', 'SepNodule_2', 'SepNodule_9',
                      'Laterality_1', 'Laterality_2', 'Laterality_3', 'Laterality_9', 
                      'PTHLTYPE_Histology_small cell carcinoma', 'PTHLTYPE_Histology_squamous cell carcinoma',
                      'PTHLTYPE_Histology_large cell carcinoma', 'PTHLTYPE_Histology_Sarcomas and soft tissue tumors',
                      'PTHLTYPE_Histology_Non-lung cancer', 'PTHLTYPE_Histology_other specified carcinoma',
                      'PTHLTYPE_Histology_Unspecified carcinoma', 'PTHLTYPE_Histology_adenocarcinoma']

# Ensure column order consistency
global_feature_en = ['Age', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 'Chemotherapy', 'Surgery',
                     'PleuInva_1', 'PleuInva_2', 'PleuInva_9', 'Gender_1', 'Gender_2',
                     'SepNodule_1', 'SepNodule_2', 'SepNodule_9',
                      'Laterality_1', 'Laterality_2', 'Laterality_3', 'Laterality_9', 
                      'PTHLTYPE_Histology_small cell carcinoma', 'PTHLTYPE_Histology_squamous cell carcinoma',
                      'PTHLTYPE_Histology_large cell carcinoma', 'PTHLTYPE_Histology_Sarcomas and soft tissue tumors',
                      'PTHLTYPE_Histology_Non-lung cancer', 'PTHLTYPE_Histology_other specified carcinoma',
                      'PTHLTYPE_Histology_Unspecified carcinoma', 'PTHLTYPE_Histology_adenocarcinoma']


taiwan_feature = ['PleuEffu', 'EGFR', 'ALK', 'MAGN', 'DIFF', 'BMI_label', 'CIG', 'BN', 'ALC']
seer_feature = ['Income', 'Area', 'Race']

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def globalFeatureDataPreprocess(x, y):
    columns_exclude = ['Age', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 'Chemotherapy', 'Surgery']
    
    # One Hot Encoding 
    x = x[global_feature] 
    x = pd.get_dummies(x, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])

    for col in one_hot_feature: 
        if col not in x.columns:
            x[col] = False

    # Ensure column order consistency
    x = x[global_feature_en]

    x, y = torch.from_numpy(x.values.astype(np.float32)).to(DEVICE), torch.from_numpy(y.astype(np.float32)).to(DEVICE)

    return x, y 


def localFeatureDataPreprocess(x, y, testset_coding_book, institution): 
    col_exclude_tw = ['Age', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 'Chemotherapy', 'Surgery', 'DIFF', 'BMI_label']
    col_exclude_seer = ['Age', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 'Chemotherapy', 'Surgery', 'Income']

    # In this experiment, local feature group contain global feature group 
    local_feature = list(global_feature)
    local_feature += (list(taiwan_feature) if institution == 1 else list(seer_feature))

    columns_exclude = list(col_exclude_tw) if institution == 1 else list(col_exclude_seer)

    x = x[local_feature]

    coding_book = []

    for col in local_feature:
        if col not in columns_exclude: 
            # train set encoding 
            if testset_coding_book is None:
                x[col], info = utils.targetEncode(x[col], y[:, 1]) 
                coding_book.append(info) 
            # test set encoding 
            else: 
                for dictt in testset_coding_book:
                    if dictt["feature_name"] == x[col].name:
                        x[col] = x[col].map(dictt["mapping"]).fillna(dictt["default"])
        else:     
            x[col] = utils.scaling(x[col])
        
    x_tensor, y_tensor = torch.from_numpy(x.values).float().to(DEVICE), torch.from_numpy(y).float().to(DEVICE)

    if testset_coding_book is None: 
        return x_tensor, y_tensor, coding_book
    else: 
        return x_tensor, y_tensor
    


class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.LayerNorm(12)
        self.fc3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(12, 6)
        self.fc5 = nn.Linear(6, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x

    def penultimateLayerOutput(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(self.fc4(x))
        return x
    

def train(net, x_train, y_train, epochs: int, class_weights, best_params, verbose=True):
    """Train the network on the training set.""" 
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
    optimizer = torch.optim.AdamW(net.parameters(), weight_decay=best_params['weight_decay'], lr=best_params['learning_rate']) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=3e-5)

    losses, auroces = [], []

    net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()   # 1. Zero out old gradients 
        outputs = net.forward(x_train) # 2. Foward pass 
        loss = criterion(outputs.float().to(DEVICE), y_train.float())  # 3. Compute loss
        loss.backward() # 4. Backward pass (compute gradients)
        optimizer.step() # 5. Update weights
        scheduler.step(loss.item()) # 6. Update learning rate

        auroc = roc_auc_score(y_true=y_train[:, 1].detach().cpu().numpy() , y_score=outputs[:, 1].detach().cpu().numpy())

        losses.append(loss.item())
        auroces.append(auroc)

        # Metrics
        if verbose:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f} | Trainset AUROC: {auroc:.4f}")

    for param_group in optimizer.param_groups:
        print("Learning Rate:", param_group['lr'])

    return losses[-1], auroces[-1] # server will only use for logging 


# Evaluate Function & Extract the Penultimate Output 
def evalPenultimate(net, x_remediation, y_remediation, x_test, y_test, coding_book, institution):

    # Federated Learning use one hot encoding, which doesn't need a coding book 
    if coding_book is None:
        x_remediation, y_remediation = globalFeatureDataPreprocess(x_remediation, y_remediation)
        x_test, y_test = globalFeatureDataPreprocess(x_test, y_test)
    else:
        x_remediation, y_remediation = localFeatureDataPreprocess(x_remediation, y_remediation, coding_book, institution) 
        x_test, y_test = localFeatureDataPreprocess(x_test, y_test, coding_book, institution)

    # Train set for second stage training 
    net.eval()
    with torch.no_grad():
        prob_remediation = net.forward(x_remediation)
        penultimate_remediation = net.penultimateLayerOutput(x_remediation)

    # Test set for testing
    net.eval()
    with torch.no_grad():
        prob_test = net.forward(x_test)
        penultimate_test = net.penultimateLayerOutput(x_test)
        auroc = roc_auc_score(y_true=y_test[:, 1].cpu().numpy(), y_score=prob_test[:, 1].cpu().numpy())
        auprc = average_precision_score(y_true=y_test[:, 1].cpu().numpy(), y_score=prob_test[:, 1].cpu().numpy())

    return auroc, auprc, penultimate_remediation, prob_remediation, penultimate_test, prob_test


def averageResult(prob_global, prob_local, y_test):
    prob_avg = 0.5 * prob_global + 0.5 * prob_local
    auroc = roc_auc_score(y_true=y_test[:, 1], y_score=prob_avg[:, 1])
    auprc = average_precision_score(y_true=y_test[:, 1], y_score=prob_avg[:, 1])

    return auroc, auprc


# Optuna Objective Function 
def objective(trial, x_train, y_train, class_weights, institution):
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)

    x_train, y_train, _ = localFeatureDataPreprocess(x_train, y_train, None, institution)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    aurocs = []

    for train_idx, val_idx in skf.split(x_train.detach().cpu().numpy(), y_train[:,1].detach().cpu().numpy()):

        x_tr, x_val = x_train[train_idx], x_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = Net(input_size=x_train.shape[1], output_size=2).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-20)

        model.train()
        for epoch in range(300):  
            optimizer.zero_grad()
            outputs = model.forward(x_tr)
            loss = criterion(outputs, y_tr)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item()) 

        model.eval()
        with torch.no_grad():
            outputs = model(x_val)

        aurocs.append(roc_auc_score(y_true=y_val[:, 1].cpu().numpy(), y_score=outputs[:, 1].cpu().numpy()))
    
    return np.mean(aurocs)


# Model definition
def federatedLearning(x_train, y_train, institution, class_weights, best_params, seed):
    x_train, y_train = globalFeatureDataPreprocess(x_train, y_train)
    FederatedNet = Net(input_size=x_train.shape[1], output_size=2).to(DEVICE)

    client_hospital = utils.SpcancerClient(FederatedNet, x_train, y_train, class_weights, best_params)
    fl.client.start_client(server_address="127.0.0.1:6000", client=client_hospital)

    # Passing seed from main is only used in here
    # utils.featureInterpreter('Localized Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return FederatedNet


def localizedLearning(x_train, y_train, institution, class_weights, best_params, seed):
    x_train, y_train, coding_book = localFeatureDataPreprocess(x_train, y_train, None, institution)
    LocalizedNet = Net(input_size=x_train.shape[1], output_size=2).to(DEVICE)

    train(LocalizedNet, x_train, y_train, epochs=1500, class_weights=class_weights, best_params=best_params)

    # Passing seed from main is only used in here
    # utils.featureInterpreter('Localized Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return LocalizedNet, coding_book


def saveEmbeddings(pen_global, prob_global, pen_local, prob_local, y):

    pen_global, prob_global = pen_global.cpu().numpy(), prob_global.cpu().numpy()
    pen_local, prob_local = pen_local.cpu().numpy(), prob_local.cpu().numpy()


    # Extract the float values from np.float32 objects
    global_features = [[float(val) for val in pen_global_row] for pen_global_row in pen_global]
    global_probs = [[float(val) for val in prob_global_row] for prob_global_row in prob_global]
    local_features = [[float(val) for val in pen_local_row] for pen_local_row in pen_local]
    local_probs = [[float(val) for val in prob_local_row] for prob_local_row in prob_local]
    targets = [y.iloc[i] for i in range(len(y))]


    # Create DataFrame
    df = pd.DataFrame({
        'Global': global_features,
        'Global_Prob': global_probs,
        'Local': local_features,
        'Local_Prob': local_probs,
        'Target': targets
    })

    return df


def main() -> None:  
    '''
    If you use the script to run this program, where you can test multiple seeds per time. You need to comment 
    LINE: institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 42
    Otherwise, you need to comment the following line, so you can only test for one seed.
    LINE: seed, institution = utils.parse_argument_for_running_script()
    '''
    institution, seed = utils.parse_argument_for_running_script()
    # institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 22

    columns = list(global_feature)

    if institution == 1:
        columns.extend(taiwan_feature)
        df = pd.read_csv('Data_folder/Taiwan_en.csv')
    else: 
        columns.extend(seer_feature)
        df = pd.read_csv('Data_folder/SEER_en.csv')

    hospital = 'Taiwan' if institution == 1 else 'USA'

    columns.append('Target')
    df = df[columns]

    trainset, testset = train_test_split(df, test_size=0.3, stratify=df['Target'], random_state=seed)
    x_train, y_train = trainset.drop(columns=['Target']), trainset['Target']
    x_test, y_test = testset.drop(columns=['Target']), testset['Target']
    y_train_one_hot = utils.to_categorical(y_train, num_classes=2)
    y_test_one_hot = utils.to_categorical(y_test, num_classes=2)


    print(f'------------------------{f"Name of your Institution: {hospital}"}------------------------')
    # class weights
    beta = (len(trainset)-1)/len(trainset)
    class_weights = utils.get_class_balanced_weights(y_train, beta)


    # Hyperparamter Optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
    opt_objective = partial(objective, x_train=x_train, y_train=y_train_one_hot, class_weights=class_weights, institution=institution)
    study.optimize(opt_objective, n_trials=100)
    best_params = study.best_params 


    FederatedNet = federatedLearning(x_train, y_train_one_hot, institution, class_weights, best_params, seed)
    LocalizedNet, coding_book = localizedLearning(x_train, y_train_one_hot, institution, class_weights, best_params, seed)

    # Extract Penultimate Layer output and evaluate the model 
    auroc_global, auprc_global, pen_train_global, prob_train_global, pen_test_global, prob_test_global = evalPenultimate(FederatedNet, x_train, y_train_one_hot, x_test, y_test_one_hot, None, -1)
    auroc_local, auprc_local, pen_train_local, prob_train_local, pen_test_local, prob_test_local = evalPenultimate(LocalizedNet, x_train, y_train_one_hot, x_test, y_test_one_hot, coding_book, institution)

    # Save Train set Embeddings into csv file 
    df = saveEmbeddings(pen_train_global, prob_train_global, pen_train_local, prob_train_local, y_train)
    df.to_csv(f"Middle/Train_{institution}.csv", index=False)

    # Save Test set into csv file 
    df = saveEmbeddings(pen_test_global, prob_test_global, pen_test_local, prob_test_local, y_test)
    df.to_csv(f"Middle/Test_{institution}.csv", index=False)

    # Late Fusion (average results from two models)
    auroc_avg, auprc_avg = averageResult(prob_test_global.cpu(), prob_test_local.cpu(), y_test_one_hot)
    print(f"Late Fusion AUROC {auroc_avg}")

    # Saving Baseline Models Results 
    baseline = {
        f'Model | {hospital} | seed={seed}': ['Federated Learning', 'Localized Learning', 'Late Fusion'],
        'auroc': [np.array(auroc_global).astype(float), np.array(auroc_local).astype(float), np.array(auroc_avg).astype(float)], 
        'auprc': [np.array(auprc_global).astype(float), np.array(auprc_local).astype(float), np.array(auprc_avg).astype(float)]
    }
    
    baseline_results = pd.DataFrame(baseline)
    baseline_results.to_csv(f'Results/Baseline_{hospital}.csv', mode='a', index=False)
        
    print("Results saved to Baseline.csv")


if __name__ == "__main__":
    main()
