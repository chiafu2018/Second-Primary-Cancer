'''
This code is for centralized learning in cross institution. It is used to compare with the algorithm we proposed. 
'''

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, roc_curve
import cen_utils
import shap

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

global_feature = ['Laterality', 'Age', 'Gender', 'SepNodule', 'PleuInva', 'Tumorsz', 'LYMND', 'AJCC', 'Radiation', 
                    'Chemotherapy', 'Surgery']

taiwan_feature = ['PleuEffu', 'EGFR', 'ALK', 'MAGN', 'DIFF', 'BMI_label', 'CIG', 'BN', 'ALC']

seer_feature = ['Income', 'Area', 'Race']

# All columns you want for training (cen+fed)
columns = list(global_feature)
institution = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))

if institution == 1:
    columns.extend(taiwan_feature)
    df = pd.read_csv('..\Data_folder\Taiwan_en.csv')
else: 
    columns.extend(seer_feature)
    df = pd.read_csv('..\Data_folder\SEER_en.csv')

columns.append('Target')
df = df[columns]

auroc = []


for seed in range(10, 15):

      trainset, testset = cen_utils.onehot_encoding(df, seed)
      x_train, y_train = trainset.drop(columns=['Target']), trainset['Target']
      x_test, y_test = testset.drop(columns=['Target']),testset['Target']

      # print(f"x_train (data number, feature number): {x_train.shape}")
      # print(f"x_test (data number, feature number): {x_test.shape}")

      opt_adam = Adam(learning_rate = 0.003)
      model = Sequential() 
      model.add(Dense(12, activation = 'relu', input_shape = (x_train.shape[1],))) 
      model.add(BatchNormalization())
      model.add(Dense(6, activation = 'sigmoid'))
      model.add(BatchNormalization())
      model.add(Dropout(0.2))
      model.add(Dense(2, activation = 'softmax'))
      model.compile(optimizer = opt_adam, loss = "categorical_crossentropy", metrics = ['accuracy'])

      y_train_one_hot = to_categorical(y_train, num_classes=2)
      y_test_one_hot = to_categorical(y_test, num_classes=2)
      
      # Choose a value for beta, e.g., 0.999 or tune it based on your dataset
      beta = 0.999
      class_weights = cen_utils.get_class_balanced_weights(y_train, beta)
      lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0005)

      history = model.fit(x_train, y_train_one_hot, epochs = 300, class_weight = class_weights, callbacks=[lr_scheduler])

      # Plot loss values during iterration 
      plt.plot(history.history['loss'])
      plt.title('Model loss -- softmax')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['Train'], loc = 'upper left')
      plt.show()
      # model.summary()

      pred_prob = model.predict(x_test)
      auroc.append(roc_auc_score(y_test, pred_prob[:, 1]))

      # Avoid predicting all zeros 
      # if np.sum(y_pred) == 0: 
      #       print("Your model just predict all zeros")
      #       exit(0)

      # Model Evaluation
      # accuracy = accuracy_score(y_test, y_pred)
      # f1 = f1_score(y_test, y_pred)
      # precision = precision_score(y_test, y_pred)
      # recall = recall_score(y_test, y_pred)
      # mcc = matthews_corrcoef(y_test, y_pred)
      # auc = roc_auc_score(y_test, pred_prob)

      # print("------------------------------- Result -------------------------------")
      # result = pd.DataFrame({'accuracy':[accuracy], 'f1': [f1], 'precision':[precision], 'recall' : [recall], 'mcc': [mcc], 'auc': [auc] })
      # print(result.to_string(index = False, float_format = '%.6f'))

      # if int(input("Do you want to save the model? (1 for yes): ")) == 1:
      #       model.save("Model")


print("------------------------------- Result -------------------------------")
print(f"avg auroc: {np.average(auroc)}")



'''     
# Shap Explainer
background_data = shap.sample(x_train, 300)  
explainer = shap.KernelExplainer(model, background_data, link = 'logit')
# explainer = shap.KernelExplainer(model, background_data, link='logit', force_cpu = True)

# Evaluate Single data
shap_values_single = explainer.shap_values(x_train.iloc[299])
print(f"Shap value: (feature number: {len(shap_values_single[0])})")
for index, val in enumerate(shap_values_single[0]):
      print(f"{x_train.iloc[299].index[index]} : {val}")

# Evaluate Multiple data
shap_values_multi = explainer.shap_values(x_train.iloc[299:305,:])
print(f"Shap value: (Mulitple cases avg)(feature number: {len(shap_values_multi[0][0])})")
for index, val in enumerate(np.mean(shap_values_multi[0], axis = 0)):
      print(f"{x_train.iloc[299].index[index]} : {val}")
'''