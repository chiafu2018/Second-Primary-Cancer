'''
This code serves as a backup in case fine-tuning fails. 
'''

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, roc_curve
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, GlobalMaxPooling1D 
from tensorflow.keras.optimizers import Adamax


# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass
    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Implement the seesawing weights algorithm 
class SeeSawingWeights(Classifier):
    def __init__(self, epoch, f_global, f_local):
        self.model = None
        self.epoch = epoch
        self.ser_weight = 0.0
        self.loc_weight = 0.0
        self.f_global = f_global
        self.f_local = f_local


    def fit(self, X, y):
        # You have to reinitialize these weights before training a new model 
        lr = 1/len(X)
        self.ser_weight = self.f_global / (self.f_global + self.f_local)
        self.loc_weight = self.f_local / (self.f_global + self.f_local)

        print("ser ini weights:", self.ser_weight)
        print("loc ini weights:", self.loc_weight)


        for cur in range(self.epoch):
            lr *= math.exp(-cur)    # let the learning rate declay

            test_array = []

            total_error = 0
            how_many_case_wrong = 0


            # Check the loss function 
            for index, row in X.iterrows():
                # total_error+=np.abs(self.ser_weight*row.iloc[0]+self.loc_weight*row.iloc[2]-y[index])
                
                yes_prob = self.ser_weight*row.iloc[0]+self.loc_weight*row.iloc[2]
                no_prob = self.ser_weight*row.iloc[1]+self.loc_weight*row.iloc[3]

                if (yes_prob>=no_prob)and(y[index]==0):
                    how_many_case_wrong+=1
                elif (yes_prob<no_prob)and(y[index]==1):
                    how_many_case_wrong+=1

            # print("Loss:", total_error)
            print("How many cases wrong:", how_many_case_wrong)



            for index, row in X.iterrows():
                
                yes_prob = self.ser_weight*row.iloc[0]+self.loc_weight*row.iloc[2]
                no_prob = self.ser_weight*row.iloc[1]+self.loc_weight*row.iloc[3]

                if ((yes_prob>=no_prob)and(y[index]==0)) or ((yes_prob<no_prob)and(y[index]==1)):

                    cg = row.iloc[0] if y[index] == 1 else row.iloc[1]
                    cl = row.iloc[2] if y[index] == 1 else row.iloc[3]

                    # correct: 0; wrong: 1
                    epsilon_global = math.ceil(max(row.iloc[0], row.iloc[1]) - cg)
                    epsilon_local = math.ceil(max(row.iloc[2], row.iloc[3]) - cl)

                    # simultaneously correct or wrong: 0; else: 1 
                    epsilon = epsilon_global*(1-epsilon_local) + epsilon_local*(1-epsilon_global)

                    delta_weights = lr*((1-epsilon)*math.exp(abs(cl-cg)/2)+epsilon*math.exp(abs(cl+cg)/2))

                    test_array.append(delta_weights)

                    if epsilon_global == 0 and epsilon_local == 0: 
                        if cl > cg:
                            self.ser_weight -= delta_weights
                            self.loc_weight += delta_weights
                        else:
                            self.ser_weight += delta_weights
                            self.loc_weight -= delta_weights
                    elif epsilon_global == 1 and epsilon_local == 1:
                        if cl < cg:
                            self.ser_weight -= delta_weights
                            self.loc_weight += delta_weights
                        else: 
                            self.ser_weight += delta_weights
                            self.loc_weight -= delta_weights

                    elif epsilon_global == 0 and epsilon_local == 1:
                        self.ser_weight += delta_weights
                        self.loc_weight -= delta_weights

                    elif epsilon_global == 1 and epsilon_local == 0:
                        self.ser_weight -= delta_weights
                        self.loc_weight += delta_weights
                
            print(f"round{cur} delta weights:", np.average(test_array))
            
                
        print("New server weights:", self.ser_weight)
        print("New local weights:", self.loc_weight)

        


    def predict(self, X, y_test):
        y = []
        pred_prob = []
        for index, row in X.iterrows():
            yes_prob = self.ser_weight*row.iloc[0]+self.loc_weight*row.iloc[2]
            no_prob = self.ser_weight*row.iloc[1]+self.loc_weight*row.iloc[3]
            pred_prob.append(yes_prob)


        fpr, tpr, threshold = roc_curve(y_test, pred_prob)
        optimal_index1 = np.argmax(tpr-fpr)
        y = [1 if (prob >= threshold[optimal_index1]).any() else 0 for prob in pred_prob]

            # if (yes_prob >= no_prob).any():
            #     y.append(np.int64(1))
            # else:
            #     y.append(np.int64(0))

        return y

    def predict_proba(self, X):
        prob = []
        for index, row in X.iterrows():
            yes_prob = self.ser_weight*row.iloc[0]+self.loc_weight*row.iloc[2]
            no_prob = self.ser_weight*row.iloc[1]+self.loc_weight*row.iloc[3]
            prob.append(np.array(yes_prob , no_prob))

        prob = np.array(prob)
        return prob
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class NeuralNetwork(Classifier):
    def __init__(self, epoch, learning_rate):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(4,), activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(4, activation='relu'))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def fit(self, X, y):
        class_wegiths = {0: 1, 1: 6}
        self.model.fit(X, y, epochs = self.epoch, class_weight = class_wegiths)


    def predict(self, X, y_test):
        pred_prob = self.model.predict(X)
        fpr, tpr, threshold = roc_curve(y_test, pred_prob)
        optimal_index1 = np.argmax(tpr-fpr)
        y = [1 if (prob >= threshold[optimal_index1]).any() else 0 for prob in pred_prob]

        return y

    
    def predict_proba(self, X):
        prob = self.model.predict(X)
        return prob


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Function to evaluate the performance of the model
def evaluate_model(model, X_test, y_test):
    # Predict using the model and calculate various performance metrics
    predictions = model.predict(X_test, y_test) #　--> y_test is for generating thresholds
    proba = model.predict_proba(X_test)

    
    # To make sure that there are two classes in test set 
    if np.sum(y_test):  
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions) 
        recall = recall_score(y_test, predictions)
        mcc = matthews_corrcoef(y_test, predictions)    
        auc = roc_auc_score(y_test, proba)
    else:
        accuracy, f1, precision, recall, mcc, auc = None, None, None, None, None, None

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }
 
# Main function to execute the pipeline
def main():

    institution = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))
    
    # Load trainWithLable data
    df = pd.read_csv(f'middle_{institution}.csv')

    X_train, y_train = df.drop('Outcome', axis=1), df['Outcome']

    # Load f-score to initialize weights(iloc is to make the val be scalar)
    df_init = pd.read_csv(f'init_{institution}.csv')
    f_global = df_init['f1 global'].iloc[0]
    f_local = df_init['f1 local'].iloc[0]

    # Define models for classification
    models = { 
                'SSW': SeeSawingWeights(epoch=40, f_global = f_global, f_local = f_local), 
                'NNs': NeuralNetwork(epoch = 20, learning_rate=0.03) 
            }

    # Perform K-Fold cross-validation
    kf = KFold(n_splits=4, random_state=42, shuffle=True)
    cv_results = []

    for name, model in models.items():
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
            print("Fold index: ", fold_idx)
            print("Number of training data:", len(X_train))

            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model.fit(X_train_fold, y_train_fold)
            fold_result = evaluate_model(model, X_val_fold, y_val_fold)
            fold_result['model'] = name
            fold_result['fold'] = fold_idx
            cv_results.append(fold_result)

    # Convert CV results to a DataFrame and calculate averages
    cv_results_df = pd.DataFrame(cv_results)
    avg_results = cv_results_df.groupby('model').mean().reset_index()
    avg_results['model'] += ' Average'
    all_results_df = pd.concat([cv_results_df, avg_results], ignore_index=True)

    # Adjust column order and display results
    # all_results_df = all_results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'auc']]
    all_results_df = all_results_df[['auc']]


    print("Cross-validation results:")
    print(all_results_df)

    # Save results to an Excel file
    all_results_df.to_csv('Results/cv_results.csv', index=False)
    print("Cross-validation results with averages saved to cv_results.csv")


if __name__ == "__main__":
    main()
