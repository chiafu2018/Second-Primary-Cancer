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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import utils1
import shap

tf.debugging.set_log_device_placement(False)

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


def federated_learning(x_train, y_train, x_test, y_test, institution, class_weights):

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
    opt_adam = Adam(learning_rate = 0.003)
    model = Sequential() 
    model.add(Dense(12, activation = 'relu', input_shape = (x_train.shape[1],))) 
    model.add(BatchNormalization())
    model.add(Dense(6, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(optimizer = opt_adam, loss = "categorical_crossentropy", metrics = ['accuracy'])

    # Start Flower client
    client_hospital = utils1.SpcancerClient(model, x_train, y_train, x_test, y_test, class_weights)
    fl.client.start_numpy_client("127.0.0.1:6000", client=client_hospital)

    pred_prob = model.predict(x_test.astype(float))
    auc = roc_auc_score(y_test, pred_prob[:, 1])
    print(f"Global model auc score: {auc}")

    # featureIntepreter(model, x_train.astype(np.int32), institution)

    return auc, pred_prob


def centralized_learning(x_train, y_train, x_test, y_test, institution, class_weights):
 
    col_exclude_tw = ['Radiation', 'Chemotherapy', 'Surgery']
    col_exclude_seer = ['Radiation', 'Chemotherapy', 'Surgery']

    local_feature = list(global_feature)
    local_feature += (list(taiwan_feature) if institution == 1 else list(seer_feature))

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
    opt_adam = Adam(learning_rate = 0.003)
    model = Sequential() 
    model.add(Dense(12, activation = 'relu', input_shape = (x_train.shape[1],))) 
    model.add(BatchNormalization())
    model.add(Dense(6, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(optimizer = opt_adam, loss = "categorical_crossentropy", metrics = ['accuracy'])

    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000005)
    history = model.fit(x_train.astype(float), y_train, epochs = 300, class_weight = class_weights, callbacks=[lr_scheduler])

    plt.plot(history.history['loss'])
    plt.title('Model loss -- centralized learning')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc = 'upper left')
    plt.show()

    pred_prob = model.predict(x_test.astype(float))
    auc = roc_auc_score(y_test, pred_prob[:, 1])
    print(f"Local model AUC score: {auc}")

    # featureIntepreter(model, x_train.astype(np.int32), institution)

    return auc, pred_prob


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
    y_train_one_hot = to_categorical(y_train, num_classes=2)


    print(f'------------------------{f"Name of your Institution: {institution}"}------------------------')
    print(f"x_train (data number, feature number): {x_train.shape}")
    print(f"x_test (data number, feature number): {x_test.shape}")
    print(f'The number of true cases in y_train:  {(y_train == 1).sum()}')
    print(f'The number of true cases in y_test:  {(y_test == 1).sum()}')

    # class weights
    beta = 0.999
    class_weights = utils1.get_class_balanced_weights(y_train, beta)

    auc_global, fed_prob = federated_learning(x_train, y_train_one_hot, x_test, y_test, institution, class_weights)
    auc_local, cen_prob = centralized_learning(x_train, y_train_one_hot, x_test, y_test, institution, class_weights)


    auc = {
        'global auc': np.array(auc_global).astype(float),
        'local auc': np.array(auc_local).astype(float)
    }

    auc_csv = pd.DataFrame(auc, index=[0])
    print(auc_csv)
    auc_csv.to_csv(f"init_{institution}.csv",index=False)

    result = {
        'global model predict yes prob': fed_prob[:,1],
        'global model predict no prob': fed_prob[:,0],
        'local model predict yes prob': cen_prob[:,1],
        'local model predict no prob': cen_prob[:,0],
        'Outcome': y_test
    }

    middle = pd.DataFrame(result)
    middle.to_csv(f"middle_{institution}.csv",index=False)



if __name__ == "__main__":
    main()
