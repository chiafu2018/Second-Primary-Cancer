import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
import optuna 
from sklearn.model_selection import cross_val_score
from functools import partial
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, average_precision_score
import utils

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"PyTorch {torch.__version__}")


def objective(trial, x_train, y_train, sample_weights):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 2, 16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(x_train, y_train):
        x_tr, x_val = x_train[train_idx], x_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = sample_weights[train_idx] 

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric='logloss'
        )

        model.fit(x_tr, y_tr, sample_weight=w_tr)

        y_val_prob = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_prob)
        aucs.append(auc)

    return np.mean(aucs)


def evaluateModel(model, x_test, y_test):
    y_prob = model.predict_proba(x_test)[:, 1]

    if (y_test).sum():
        auprc = average_precision_score(y_test, y_score=y_prob)
        auroc = roc_auc_score(y_test, y_score=y_prob)
    else:
        auprc, auroc = None, None

    return {
        'auprc': auprc, 
        'auroc': auroc
    }

def main():
    '''
    If you use the script to run this program, where you can test multiple seeds per time. You need to comment 
    LINE: institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): "))
    Otherwise, you need to comment, where you can only test for one seed.
    LINE: institution, seed = utils.parse_argument_for_running_script()
    '''
    institution, seed = utils.parse_argument_for_running_script()
    # institution, seed = int(input("Please choose a hospital: 1 for Taiwan, 2 for US (SEER Database): ")), 22
    
    df_train = pd.read_csv(f'Middle/Train_{institution}.csv')
    trainset = utils.csvPreprocess(df_train)

    df_test = pd.read_csv(f'Middle/Test_{institution}.csv')
    testset = utils.csvPreprocess(df_test)

    x_train, y_train = trainset.drop(columns=['Target']).values, trainset['Target'].values
    x_test, y_test = testset.drop(columns=['Target']).values, testset['Target'].values


    # class weights
    beta = (len(df_train['Target'])-1)/len(df_train['Target'])
    class_weights = utils.get_class_balanced_weights(df_train['Target'], beta)
    class_weights = {i: w for i, w in enumerate(class_weights.cpu().tolist())}
    sample_weights = np.array([class_weights[y] for y in y_train])

    # Hyperparameter Optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
    opt_objective = partial(objective, x_train=x_train, y_train=y_train, sample_weights=sample_weights)
    study.optimize(opt_objective, n_trials=100)
    best_params = study.best_params 

    # Retrain model with best hyperparameter 
    model = XGBClassifier(n_estimators=best_params['n_estimators'], 
                          max_depth=best_params['max_depth'],
                          learning_rate=best_params['learning_rate'])

    model.fit(x_train, y_train, sample_weight=sample_weights)
    
    metrics_results = evaluateModel(model, x_test, y_test)

    results = pd.DataFrame([{
        'Seed': seed,
        'AUROC': metrics_results['auroc'],
        'AUPRC': metrics_results['auprc']
    }])

    hospital = 'Taiwan' if institution == 1 else 'USA'

    results.to_csv(f'Results/XGB_{hospital}.csv', mode='a', index=False)
    print("results saved to XGB.csv")


if __name__ == "__main__":
    main()
