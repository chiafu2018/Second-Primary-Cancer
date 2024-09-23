import numpy as np
import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import train_test_split


def onehot_encoding(df, seed):
    train = pd.DataFrame()
    validation = pd.DataFrame()
    test = pd.DataFrame()
    columns_exclude = ['Radiation', 'Chemotherapy', 'Surgery', 'Target'] 
    df = pd.get_dummies(df, drop_first=False, columns=[col for col in df.columns if col not in columns_exclude])

    trainset, testset = train_test_split(df, test_size = 0.1, stratify=df['Target'], random_state = seed)
    train = pd.concat([train, trainset])
    test = pd.concat([test, testset])
    trainset = trainset.astype(int)
    testset = testset.astype(int)
    return trainset, testset

def get_class_balanced_weights(y_train, beta):
    # Count the number of samples for each class
    class_counts = Counter(y_train)
    total_samples = len(y_train)

    # Calculate the effective number for each class
    effective_num = {}
    for class_label, count in class_counts.items():
        effective_num[class_label] = (1 - beta**count) / (1 - beta)
    
    # Calculate the class-balanced weight
    class_weights = {class_label: total_samples / (len(class_counts) * effective_num[class_label]) for class_label in class_counts}
    print(class_weights)
    return class_weights