'''
This code serves as a backup in case fine-tuning fails. 
'''

import os
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import  fbeta_score, roc_curve, roc_auc_score
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''''''''''''''''''''''''''''''''' Feature Groups '''''''''''''''''''''''''''''''''''''''

global_feature = ['Laterality', 'Age', 'Gender', 'SepNodule', 'PleuInva', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 
                    'Chemotherapy', 'Surgery']

# Even in global feature group every institutions have same features, the elements inside after doing one hot encoding may be 
# different. Hence, we need to use global_feature_en to unify all the features. 
global_feature_en = ['Age_6', 'Tumorsz_1', 'Tumorsz_4', 'LYMND_3', 'Chemotherapy_1', 'AJCC_1', 'Surgery_2', 
                    'SepNodule_2', 'Laterality_2', 'PleuInva_1', 'Tumorsz_2', 'AJCC_3', 'Laterality_1', 'Age_4',
                    'Chemotherapy_2', 'LYMND_9', 'Gender_2', 'Tumorsz_9', 'Age_7', 'Age_9', 'Gender_1', 'AJCC_2',
                    'Laterality_3', 'Radiation_1', 'Laterality_9', 'LYMND_5', 'Age_3', 'PleuInva_9', 'Radiation_2',
                    'Tumorsz_3', 'LYMND_1', 'LYMND_4', 'Age_2', 'AJCC_5', 'Age_8', 'AJCC_9', 'AJCC_4', 'PleuInva_2',
                    'LYMND_2', 'Surgery_1', 'Age_5', 'SepNodule_9', 'SepNodule_1']

taiwan_feature = ['PleuEffu', 'EGFR', 'ALK', 'MAGN', 'DIFF', 'BMI_label', 'CIG', 'BN', 'ALC']

seer_feature = ['Income', 'Area', 'Race']

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def federated_learning(x_train, y_train, x_test, y_test):

    x_train = x_train[global_feature]
    x_test = x_test[global_feature]

    # One hot encoding 
    columns_exclude = ['Radiation', 'Chemotherapy', 'Surgery']

    dfencode = pd.DataFrame()
    dfencode = pd.concat([dfencode, x_train])
    x_train = pd.get_dummies(dfencode, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])

    for col in global_feature_en: 
        if col not in x_train.columns:
            x_train[col] = 0

    dfencode = pd.DataFrame()
    dfencode = pd.concat([dfencode, x_test])
    x_test = pd.get_dummies(dfencode, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])

    for col in global_feature_en: 
        if col not in x_test.columns:
            x_test[col] = 0

    # Load and compile Keras model
    model = Sequential() 
    model.add(Dense(12, activation='sigmoid', input_shape=(x_train.shape[1],), name='base1'))
    model.add(Dense(6, activation='sigmoid', name='base2'))    
    model.add(Dense(1, activation='sigmoid', name='personal'))
    
    model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics=['accuracy'])


    # Start Flower client
    client_hosptial = utils.SpcancerClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("127.0.0.1:6000", client=client_hosptial)

    pred_prob = model.predict(x_test.astype(float))

    # Calculating f1_score
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    optimal_index = np.argmax(tpr-fpr)
    y_pred = (pred_prob >= threshold[optimal_index]).astype(int)
    f1 = fbeta_score(y_test, y_pred, beta=2)
    
    # model.save("Global_Model")
    print(f"Global model auc score: {roc_auc_score(y_score=y_pred, y_true=y_test)}")

    return f1, pred_prob, y_pred

def centralized_learning(x_train, y_train, x_test, y_test, institution):
 
    col_exclude_tw = []
    col_exclude_seer = []

    local_feature = list(taiwan_feature) if institution == 1 else list(seer_feature)
    columns_exclude = list(col_exclude_tw) if institution == 1 else list(col_exclude_seer)

    x_train = x_train[local_feature]
    x_test = x_test[local_feature]

    # One hot encoding 
    dfencode = pd.DataFrame()
    dfencode = pd.concat([dfencode, x_train])
    x_train = pd.get_dummies(dfencode, drop_first=False, columns=[col for col in local_feature if col not in columns_exclude])

    dfencode = pd.DataFrame()
    dfencode = pd.concat([dfencode, x_test])
    x_test = pd.get_dummies(dfencode, drop_first=False, columns=[col for col in local_feature if col not in columns_exclude])

    # Load and compile Keras model
    model = Sequential() 
    model.add(Dense(12, activation = 'sigmoid', input_shape = (x_train.shape[1],)))
    model.add(Dense(6, activation = 'sigmoid'))    
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])

    class_weights = {0: 1, 1: 5} 
    model.fit(x_train, y_train, epochs = 40, class_weight = class_weights)

    pred_prob = model.predict(x_test.astype(float))

    # Calculating f1_score
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    optimal_index = np.argmax(tpr-fpr)
    y_pred = (pred_prob >= threshold[optimal_index]).astype(int)
    f1 = fbeta_score(y_test, y_pred, beta=2)

    # model.save("Local_Model")
    print(f"Local model auc score: {roc_auc_score(y_score=y_pred, y_true=y_test)}")


    return f1, pred_prob, y_pred


def main() -> None:  
    # All columns you want for training (cen+fed)
    columns = list(global_feature)
    institution = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))

    if institution == 1:
        columns.extend(taiwan_feature)
        df = pd.read_csv('Data_folder\Taiwan_en.csv')
    else: 
        columns.extend(seer_feature)
        df = pd.read_csv('Data_folder\SEER_en.csv')

    columns.append('Target')
    df = df[columns]

    # Split it into trainset and testset (_1:first part training dataset)
    trainset, testset = train_test_split(df, test_size = 0.4, stratify = df['Target'], random_state = 42)


    x_train, y_train = trainset.drop(columns=['Target']), trainset['Target']
    x_test, y_test = testset.drop(columns=['Target']), testset['Target']


    print(f'------------------------{f"Name of your Institution: {institution}"}------------------------')
    print(f"x_train (data number, feature number): {x_train.shape}")
    print(f"x_test (data number, feature number): {x_test.shape}")
    print(f'The number of true cases in y_train:  {(y_train == 1).sum()}')
    print(f'The number of true cases in y_test:  {(y_test == 1).sum()}')
   
    f1_global, fed_prob, y_fed = federated_learning(x_train, y_train, x_test, y_test)
    f1_local, cen_prob, y_cen = centralized_learning(x_train, y_train, x_test, y_test, institution)

    # Make sure that models don't predict all zeros
    if(np.sum(y_cen)>0 and np.sum(y_fed)>0):
        f1 = {
            'f1 global': np.array(f1_global).astype(float),
            'f1 local': np.array(f1_local).astype(float)
        }
        f1_csv = pd.DataFrame(f1, index=[0])
        print(f1_csv)
        f1_csv.to_csv(f"init_{institution}.csv",index=False)

        result = {
            'global model predict yes prob': fed_prob.reshape(-1,),
            'global model predict no prob': np.array([1 - prob for prob in fed_prob.reshape(-1,)]),
            'local model predict yes prob': cen_prob.reshape(-1,),
            'local model predict no prob': np.array([1 - prob for prob in cen_prob.reshape(-1,)]),
            'Outcome': y_test
        }

        middle = pd.DataFrame(result)
        middle.to_csv(f"middle_{institution}.csv",index=False)

    else:
        print("Your model just predict all zeros. Please train both your models again or enlarge your testset")


if __name__ == "__main__":
    main()