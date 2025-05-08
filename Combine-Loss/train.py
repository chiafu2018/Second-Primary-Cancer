import flwr as fl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import optuna
from functools import partial
from sklearn.model_selection import StratifiedKFold
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

    x_tensor, y_tensor = torch.from_numpy(x.values.astype(np.float32)).to(DEVICE), torch.from_numpy(y.astype(np.float32)).to(DEVICE)

    return x_tensor, y_tensor


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
    


class federateNet(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(federateNet, self).__init__()
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
    

class pruneNet(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(pruneNet, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.softmax(self.fc1(x), dim=1)
        return x


class mainNet(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(mainNet, self).__init__()
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



def federateTrain(net, x_train, y_train, epochs: int, class_weights, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
    optimizer = torch.optim.AdamW(net.parameters(), weight_decay=1e-5, lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=3e-5)

    losses = []
    aurocs = []

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
        aurocs.append(auroc)

        # Metrics
        if verbose:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f} | Trainset AUROC {auroc:.4f}")

    for param_group in optimizer.param_groups:
        print("Learning Rate:", param_group['lr'])

    return losses[-1], aurocs[-1] # server will only use for logging 



# Optuna Objective Function 
def objective(trial,  main_x_train, prune_x_train, y_train, class_weights, institution):
    prune_wd = trial.suggest_float('prune_wd', 1e-10, 1e-3)
    prune_lr = trial.suggest_float('prune_lr', 1e-5, 1e-2)
    main_wd = trial.suggest_float('main_wd', 1e-10, 1e-3)
    main_lr = trial.suggest_float('main_lr', 1e-5, 1e-2)
    navigator = trial.suggest_float('navigator', 0, 2)

    main_x_train, y_train, _ = localFeatureDataPreprocess(main_x_train, y_train, None, institution)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    aurocs = []

    for train_idx, val_idx in skf.split(main_x_train.detach().cpu().numpy(), y_train[:,1].detach().cpu().numpy()):

        main_x_tr, main_x_val = main_x_train[train_idx], main_x_train[val_idx]
        prune_x_tr, prune_x_val = prune_x_train[train_idx], prune_x_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        PruneNet = pruneNet(input_size=6, output_size=2).to(DEVICE)
        MainNet = mainNet(input_size=main_x_train.shape[1], output_size=2).to(DEVICE)

        prune_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
        prune_optimizer = torch.optim.AdamW(PruneNet.parameters(), weight_decay=prune_wd, lr=prune_lr)
        prune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(prune_optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-15)
        
        main_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
        main_optimizer = torch.optim.AdamW(MainNet.parameters(), weight_decay=main_wd, lr=main_lr)
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(main_optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-15)

        MainNet.train()
        PruneNet.train()
        for epoch in range(1500):
            main_optimizer.zero_grad()   # 1. Zero out old gradients 
            prune_optimizer.zero_grad()

            main_outputs = MainNet.forward(main_x_tr)  # 2. Foward pass 
            prune_outputs = PruneNet.forward(prune_x_tr)

            main_loss = main_criterion(main_outputs.float().to(DEVICE), y_tr.float())  # 3. Compute loss
            prune_loss = prune_criterion(prune_outputs.float().to(DEVICE), y_tr.float())
            
            loss = main_loss + navigator * prune_loss

            loss.backward() # 4. Backward pass (compute gradients)

            main_optimizer.step() # 5. Update weights
            prune_optimizer.step() 

            main_scheduler.step(loss.item()) # 6. Update learning rate
            prune_scheduler.step(loss.item()) 

        MainNet.eval()
        PruneNet.eval()
        with torch.no_grad():
            main_outputs = MainNet.forward(main_x_val)
            prune_outputs = PruneNet.forward(prune_x_val)
        
        probability = (1 / (1 + navigator)) * main_outputs[:, 1].cpu().numpy() + (navigator / (1 + navigator)) * prune_outputs[:, 1].cpu().numpy() 

        aurocs.append(roc_auc_score(y_true=y_val[:, 1].cpu().numpy(), y_score=probability))

    return np.mean(aurocs)



def mergeTrain(MainNet, main_x_train, PruneNet, prune_x_train, y_train, epochs: int, class_weights, best_param, verbose=True):
    """Train the network on the training set."""

    navigator = best_param['navigator']

    prune_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
    prune_optimizer = torch.optim.AdamW(PruneNet.parameters(), weight_decay=best_param['prune_wd'], lr=best_param['prune_lr'])
    prune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(prune_optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-15)
    
    main_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) 
    main_optimizer = torch.optim.AdamW(MainNet.parameters(), weight_decay=best_param['main_wd'], lr=best_param['main_lr'])
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(main_optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-15)


    PruneNet.train()
    MainNet.train()
    for epoch in range(epochs):
        main_optimizer.zero_grad()   # 1. Zero out old gradients 
        prune_optimizer.zero_grad()

        main_outputs = MainNet.forward(main_x_train)  # 2. Foward pass 
        prune_outputs = PruneNet.forward(prune_x_train)

        main_loss = main_criterion(main_outputs.float().to(DEVICE), y_train.float())  # 3. Compute loss
        prune_loss = prune_criterion(prune_outputs.float().to(DEVICE), y_train.float())
        
        loss = main_loss + navigator * prune_loss

        loss.backward() # 4. Backward pass (compute gradients)

        main_optimizer.step() # 5. Update weights
        prune_optimizer.step() 

        prune_scheduler.step(loss.item()) # 6. Update learning rate
        main_scheduler.step(loss.item()) 
            
        # Metrics
        if verbose:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f}")




# Evaluate Function & Extract the Penultimate Output 
def evalPenultimate(net, x_remediation, y_remediation, x_test, y_test):

    x_remediation, y_remediation = globalFeatureDataPreprocess(x_remediation, y_remediation)
    x_test, y_test = globalFeatureDataPreprocess(x_test, y_test)


    net.eval()
    with torch.no_grad():
        prob_remediation = net.forward(x_remediation)
        penultimate_remediation = net.penultimateLayerOutput(x_remediation)

    # Testset for testing
    net.eval()
    with torch.no_grad():
        prob_test = net.forward(x_test)
        penultimate_test = net.penultimateLayerOutput(x_test)
        auroc = roc_auc_score(y_true=y_test[:, 1].cpu().numpy(), y_score=prob_test[:, 1].cpu().numpy())
        auprc = average_precision_score(y_true=y_test[:, 1].cpu().numpy(), y_score=prob_test[:, 1].cpu().numpy())

    return {"auroc":auroc, "auprc": auprc}, penultimate_remediation, prob_remediation, penultimate_test, prob_test



def federatedLearning(x_train, y_train, institution, class_weights, seed):
    x_train,  y_train = globalFeatureDataPreprocess(x_train, y_train)

    FederatedNet = federateNet(input_size=27, output_size=2).to(DEVICE)

    client_hospital = utils.SpcancerClient(FederatedNet, x_train, y_train, class_weights)
    fl.client.start_client(server_address="127.0.0.1:6000", client=client_hospital)
    
    # Passing seed from main is only used in here
    # utils.featureInterpreter('Localized Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return FederatedNet



def mergedLearning(embeddings, x_train, y_train, institution, class_weights, best_params, seed):
    x_train, y_train, coding_book = localFeatureDataPreprocess(x_train, y_train, None, institution)

    PruneNet = pruneNet(input_size=6, output_size=2).to(DEVICE)
    MainNet = mainNet(input_size=x_train.shape[1], output_size=2).to(DEVICE)

    mergeTrain(MainNet, x_train, PruneNet, embeddings, y_train, epochs=1500, class_weights=class_weights, best_param=best_params)

    # Passing seed from main is only used in here
    # utils.featureInterpreter('Localized Learning', model, x_train.astype(np.int32), institution, 'baseline', seed)

    return MainNet, PruneNet, coding_book



def evaluateModel(MainNet, PruneNet, main_x_test, prune_x_test, y_test, navigator, coding_book, institution):
    x_tensor, y_tensor= localFeatureDataPreprocess(main_x_test, y_test, coding_book, institution)
    y_test = y_tensor[:, 1].cpu().numpy()

    MainNet.eval()
    PruneNet.eval()
    with torch.no_grad():
        main_outputs = MainNet.forward(x_tensor)
        prune_outputs = PruneNet.forward(prune_x_test)
    
    probability = (1 / (1 + navigator)) * main_outputs[:, 1].cpu().numpy() + (navigator / (1 + navigator)) * prune_outputs[:, 1].cpu().numpy() 
    
    if(y_test).sum():
        auprc = average_precision_score(y_true=y_test, y_score=probability)
        auroc = roc_auc_score(y_true=y_test, y_score=probability)
    else:
        auprc, auroc = None, None

    return {
        'auprc': auprc, 
        'auroc': auroc
    }



def main() -> None:  
    '''
    If you use the script to run this program, where you can test multiple seeds per time. You need to comment 
    LINE: institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 42
    Otherwise, you need to comment the following line, so you can only test for one seed.
    LINE: seed, institution = utils.parse_argument_for_running_script()
    '''
    institution, seed = utils.parse_argument_for_running_script()
    # institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 23

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

    trainset, testset = train_test_split(df, test_size=0.1, stratify=df['Target'], random_state=seed)
    x_train, y_train = trainset.drop(columns=['Target']), trainset['Target']
    x_test, y_test = testset.drop(columns=['Target']), testset['Target']
    y_train_one_hot = utils.to_categorical(y_train, num_classes=2)
    y_test_one_hot = utils.to_categorical(y_test, num_classes=2)


    print(f'------------------------{f"Name of your Institution: {hospital}"}------------------------')
    # Class Weights
    beta = (len(trainset)-1)/len(trainset)
    class_weights = utils.get_class_balanced_weights(y_train, beta)


    FederatedNet = federatedLearning(x_train, y_train_one_hot, institution, class_weights, seed)
    metrics_baseline, embeddings, prob_train, embeddings_test, prob_test = evalPenultimate(FederatedNet, x_train, y_train_one_hot, x_test, y_test_one_hot)

    baseline = pd.DataFrame([{
        'Seed': seed,
        'AUROC': metrics_baseline['auroc'],
        'AUPRC': metrics_baseline['auprc']
    }])

    print(f"Global AUROC {metrics_baseline['auroc']} | Global AUPRC {metrics_baseline['auprc']}")
    baseline.to_csv(f'Results/Baseline_{hospital}.csv', mode='a', index=False)
    print("Results saved to Baseline.csv")

    alpha = 3
    confidence_score = np.exp(alpha * ((-prob_train.cpu()[:, 0] * prob_train.cpu()[:, 1])+0.25)).to(DEVICE)
    embeddings *= confidence_score.unsqueeze(1)
    
    confidence_score = np.exp(alpha * ((-prob_test.cpu()[:, 0] * prob_test.cpu()[:, 1])+0.25)).to(DEVICE)
    embeddings_test *= confidence_score.unsqueeze(1)


    # Hyperparamter Optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
    opt_objective = partial(objective, main_x_train=x_train, prune_x_train=embeddings, y_train=y_train_one_hot, class_weights=class_weights, institution=institution)
    study.optimize(opt_objective, n_trials=100)
    best_params = study.best_params 


    MainNet, PruneNet, coding_book = mergedLearning(embeddings, x_train, y_train_one_hot, institution, class_weights, best_params, seed)
    
    navigator = best_params['navigator']
    metrics_results = evaluateModel(MainNet, PruneNet, x_test, embeddings_test, y_test_one_hot, navigator, coding_book, institution)

    print("Cross-validation results:")

    results = pd.DataFrame([{
        'Seed': seed,
        'AUROC': metrics_results['auroc'],
        'AUPRC': metrics_results['auprc'], 
        'Beta': navigator
    }])

    results.to_csv(f'Results/Results_{hospital}.csv', mode='a', index=False)
    print("Results saved to Results.csv")

if __name__ == "__main__":
    main()
