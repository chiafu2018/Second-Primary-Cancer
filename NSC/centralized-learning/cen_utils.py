import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split


def onehot_encoding(df):
    train = pd.DataFrame()
    test = pd.DataFrame()
    columns_exclude = ['Radiation', 'Chemotherapy', 'Surgery', 'Target'] 
    df = pd.get_dummies(df, drop_first=False, columns=[col for col in df.columns if col not in columns_exclude])

    trainset, testset = train_test_split(df, test_size = 0.2, stratify=df['Target'], random_state = 42)
    train = pd.concat([train, trainset])
    test = pd.concat([test, testset])
    trainset = trainset.astype(int)
    testset = testset.astype(int)
    return trainset, testset

