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

global_feature = ['Laterality', 'Age', 'Gender', 'PleuInva', 'Tumorsz', 'LYMND', 'Radiation', 
                 'Chemotherapy', 'Surgery', 'PTHLTYPE_Histology', 'SepNodule', 'AJCC']

# Ensure different clients with same features after one hot
one_hot_feature = ['PleuInva_1', 'PleuInva_2', 'PleuInva_9', 'Gender_1', 'Gender_2',
                      'Laterality_1', 'Laterality_2', 'Laterality_3', 'Laterality_9', 
                      'SepNodule_1', 'SepNodule_2', 'SepNodule_9',
                      'PTHLTYPE_Histology_small cell carcinoma', 'PTHLTYPE_Histology_squamous cell carcinoma',
                      'PTHLTYPE_Histology_large cell carcinoma', 'PTHLTYPE_Histology_Sarcomas and soft tissue tumors',
                      'PTHLTYPE_Histology_Non-lung cancer', 'PTHLTYPE_Histology_other specified carcinoma',
                      'PTHLTYPE_Histology_Unspecified carcinoma', 'PTHLTYPE_Histology_adenocarcinoma']

# Ensure column order consistency
global_feature_en = ['Age', 'Tumorsz', 'LYMND', 'Radiation', 'Chemotherapy', 'Surgery', 'AJCC',
                     'PleuInva_1', 'PleuInva_2', 'PleuInva_9', 'Gender_1', 'Gender_2',
                      'Laterality_1', 'Laterality_2', 'Laterality_3', 'Laterality_9', 
                      'SepNodule_1', 'SepNodule_2', 'SepNodule_9',
                      'PTHLTYPE_Histology_small cell carcinoma', 'PTHLTYPE_Histology_squamous cell carcinoma',
                      'PTHLTYPE_Histology_large cell carcinoma', 'PTHLTYPE_Histology_Sarcomas and soft tissue tumors',
                      'PTHLTYPE_Histology_Non-lung cancer', 'PTHLTYPE_Histology_other specified carcinoma',
                      'PTHLTYPE_Histology_Unspecified carcinoma', 'PTHLTYPE_Histology_adenocarcinoma']


taiwan_feature = ['PleuEffu', 'EGFR', 'ALK', 'MAGN', 'DIFF', 'BMI_label', 'CIG', 'BN', 'ALC']
seer_feature = ['Income', 'Area', 'Race']

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def globalFeatureDataPreprocess(x, y):
    columns_exclude = ['Age', 'Tumorsz', 'LYMND', 'Radiation', 'Chemotherapy', 'Surgery', 'AJCC']
    
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
    col_exclude_tw = ['DIFF', 'BMI_label']
    col_exclude_seer = ['Income']

    local_feature = (list(taiwan_feature) if institution == 1 else list(seer_feature))

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
    


class SharedModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.LayerNorm(12)
        self.fc3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(12, 6)
        self.fc5 = nn.Linear(6, num_classes)


    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        x4 = F.sigmoid(self.fc4(x3))
        x5 = F.softmax(self.fc5(x4), dim=1)
        return x1, x2, x3, x4, x5
    


class PrivateModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.LayerNorm(12)
        self.fc3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(12, 6)
        self.fc5 = nn.Linear(6, num_classes)

        # Learnable Lateral Parameters 
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.alpha3 = nn.Parameter(torch.tensor(1.0))
        self.alpha4 = nn.Parameter(torch.tensor(1.0))


    def forward(self, private_input, x1, x2, x3, x4, x5):      
        x = self.fc1(private_input) + self.alpha1 * x1
        x = self.fc2(x) + self.alpha2 * x2
        x = self.fc3(x) + self.alpha3 * x3
        x = self.fc4(x) + self.alpha4 * x4
        x = F.softmax(self.fc5(x) + x5, dim=1)
        return x



def evaluateModel(shared_model:SharedModel, private_model:PrivateModel, shared_x_test, private_x_test, y_test):
    shared_model.eval()
    private_model.eval()
    with torch.no_grad():
        x1, x2, x3, x4, x5 = shared_model.forward(shared_x_test)
        outputs = private_model.forward(private_x_test, x1, x2, x3, x4, x5)
    
    y_test = y_test[:, 1].cpu().numpy()
    probability = outputs[:, 1].cpu().numpy()
    
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


    shared_x_train, y_train, = globalFeatureDataPreprocess(x_train, y_train_one_hot)
    private_x_train, y_train, coding_book = localFeatureDataPreprocess(x_train, y_train_one_hot, None, institution)


    shared_model = SharedModel(input_size=shared_x_train.shape[1], num_classes=2).to(DEVICE)
    private_model = PrivateModel(input_size=private_x_train.shape[1], num_classes=2).to(DEVICE)

    client_hospital = utils.CustomClient(shared_model, private_model, shared_x_train, private_x_train, y_train, class_weights)
    fl.client.start_client(server_address="127.0.0.1:6000", client=client_hospital)

    shared_x_test, y_test, = globalFeatureDataPreprocess(x_test, y_test_one_hot)
    private_x_test, y_test = localFeatureDataPreprocess(x_test, y_test_one_hot, coding_book, institution)

    metrics_results = evaluateModel(shared_model, private_model, shared_x_test, private_x_test, y_test)

    print("Cross-validation results:")

    results = pd.DataFrame([{
        'Seed': seed,
        'AUROC': metrics_results['auroc'], 
        'AUPRC': metrics_results['auprc'], 
    }])

    results.to_csv(f'Results/Results_{hospital}.csv', mode='a', index=False)
    print("Results saved to Results.csv")

if __name__ == "__main__":
    main()
