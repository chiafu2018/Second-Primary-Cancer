'''
You can modify the following algorithm (code) as you want. 
The old-version folder has the back-up. 
'''

import os
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import utils
import shap

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''''''''''''''''''''''''''''''''' Feature Groups '''''''''''''''''''''''''''''''''''''''

global_feature = ['Laterality', 'Age', 'Gender', 'SepNodule', 'PleuInva', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 
                 'Chemotherapy', 'Surgery']

global_feature_en = ['Age_6', 'Tumorsz_1', 'Tumorsz_4', 'LYMND_3', 'Chemotherapy_1', 'AJCC_1', 'Surgery_2', 
                     'SepNodule_2', 'Laterality_2', 'PleuInva_1', 'Tumorsz_2', 'AJCC_3', 'Laterality_1', 'Age_4',
                     'Chemotherapy_2', 'LYMND_9', 'Gender_2', 'Tumorsz_9', 'Age_7', 'Age_9', 'Gender_1', 'AJCC_2',
                     'Laterality_3', 'Radiation_1', 'Laterality_9', 'LYMND_5', 'Age_3', 'PleuInva_9', 'Radiation_2',
                     'Tumorsz_3', 'LYMND_1', 'LYMND_4', 'Age_2', 'AJCC_5', 'Age_8', 'AJCC_9', 'AJCC_4', 'PleuInva_2',
                     'LYMND_2', 'Surgery_1', 'Age_5', 'SepNodule_9', 'SepNodule_1']

taiwan_feature = ['PleuEffu', 'EGFR', 'ALK', 'MAGN', 'DIFF', 'BMI_label', 'CIG', 'BN', 'ALC']
seer_feature = ['Income', 'Area', 'Race']

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def featureIntepreter(model, x_train, institution):
    # Shap Explainer
    background_data = shap.sample(x_train, 300)  
    explainer = shap.KernelExplainer(model, background_data, link = 'logit')

    # Evaluate Single data & let feature name sort in lex order
    shap_values_single = explainer.shap_values(x_train.iloc[299])
    shap_values_single = sorted(list(zip(x_train.iloc[299].index, shap_values_single[0])))

    print(f"Shap value: (single)")
    for feature, value in shap_values_single:
        print(f"{feature} : {value}")


    # Evaluate Multiple data & average the results
    shap_values_multi = np.mean(explainer.shap_values(x_train.iloc[299:399,:]), axis=0)
    shap_values_multi = sorted(list(zip(x_train.iloc[299].index, shap_values_multi[0])))

    print(f"Shap value: (Mulitple cases average)")
    for feature, value in shap_values_multi:
        print(f"{feature} : {value}")


    # Separate the sorted pairs into two lists: features and values
    sorted_features, sorted_values = zip(*shap_values_multi)

    # Plot the sorted SHAP values into a bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features, sorted_values, color='blue')
    plt.xlabel('SHAP Value')
    plt.ylabel('Feature')

    if institution == 1:
        plt.title('SHAP Values for Taiwan Features')
    else:
        plt.title('SHAP Values for USA Features')
    plt.show()


def federated_learning(x_train, y_train, x_test, y_test, institution):

    x_train = x_train[global_feature]
    x_test = x_test[global_feature]

    # One hot encoding 
    columns_exclude = ['Radiation', 'Chemotherapy', 'Surgery']

    x_train = pd.get_dummies(x_train, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])
    x_test = pd.get_dummies(x_test, drop_first=False, columns=[col for col in global_feature if col not in columns_exclude])

    for col in global_feature_en: 
        if col not in x_train.columns:
            x_train[col] = 0
        if col not in x_test.columns:
            x_test[col] = 0

    # Ensure columns order is the same
    x_train = x_train[global_feature_en]
    x_test = x_test[global_feature_en]

    # Load and compile Keras model
    model = Sequential() 
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Start Flower client
    client_hospital = utils.SpcancerClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("127.0.0.1:6000", client=client_hospital)

    pred_prob = model.predict(x_test.astype(float))

    # Calculating f1_score
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    optimal_index = np.argmax(tpr - fpr)
    y_pred = (pred_prob >= threshold[optimal_index]).astype(int)
    f1 = fbeta_score(y_test, y_pred, beta=2)
    
    # model.save("Global_Model")
    print(f"Global model auc score: {roc_auc_score(y_score=y_pred, y_true=y_test)}")

    featureIntepreter(model, x_train.astype(np.int32), institution)

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
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))    
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    class_weights = {0: 1, 1: 10000}
    model.fit(x_train, y_train, epochs=40, class_weight=class_weights)

    pred_prob = model.predict(x_test.astype(float))

    # Calculating f1_score
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    optimal_index = np.argmax(tpr-fpr)
    y_pred = (pred_prob >= threshold[optimal_index]).astype(int)
    f1 = fbeta_score(y_test, y_pred, beta=2)

    print(f"Local model AUC score: {roc_auc_score(y_score=y_pred, y_true=y_test)}")

    featureIntepreter(model, x_train.astype(np.int32), institution)

    return f1, pred_prob, y_pred


def main() -> None:  
    columns = list(global_feature)
    institution = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))

    if institution == 1:
        columns.extend(taiwan_feature)
        df = pd.read_csv('Data_folder/Taiwan_en.csv')
    else: 
        columns.extend(seer_feature)
        df = pd.read_csv('Data_folder/SEER_en.csv')

    columns.append('Target')
    df = df[columns]

    trainset, testset = train_test_split(df, test_size=0.4, stratify=df['Target'], random_state=42)

    x_train, y_train = trainset.drop(columns=['Target']), trainset['Target']
    x_test, y_test = testset.drop(columns=['Target']), testset['Target']

    print(f'------------------------{f"Name of your Institution: {institution}"}------------------------')
    print(f"x_train (data number, feature number): {x_train.shape}")
    print(f"x_test (data number, feature number): {x_test.shape}")
    print(f'The number of true cases in y_train:  {(y_train == 1).sum()}')
    print(f'The number of true cases in y_test:  {(y_test == 1).sum()}')

    f1_global, fed_prob, y_fed = federated_learning(x_train, y_train, x_test, y_test, institution)
    f1_local, cen_prob, y_cen = centralized_learning(x_train, y_train, x_test, y_test, institution)

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
        print("Your model just predicts all zeros. Please train both your models again or enlarge your test set")


if __name__ == "__main__":
    main()
