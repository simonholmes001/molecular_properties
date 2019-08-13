import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

import os

from contextlib import contextmanager
import time
import gc

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "data")

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def import_data():

    print("[INFO]: Importing datasets...")

    df_train = pd.read_csv(os.path.join(DATA_DIR,"df_train_2.csv"))

    print("[INFO]: Datasets loaded")

    print("[INFO]: Preparting datasets...")

    df_train.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_train.fillna(0, inplace=True)

    labels = df_train.type.unique()

    dfs = {}
    for label in labels:
        dfs[label] = df_train[df_train['type'] == label]

    df1 = dfs["1JHC"]
    df2 = dfs["2JHH"]
    df3 = dfs["1JHN"]
    df4 = dfs["2JHN"]
    df5 = dfs["2JHC"]
    df6 = dfs["3JHH"]
    df7 = dfs["3JHC"]
    df8 = dfs["3JHN"]

    print("[INFO]: Preparing train_test_split...")

    y1 = pd.DataFrame(data = df1, columns=["scalar_coupling_constant"])
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df1.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y1,test_size = 0.2, random_state=10)

    y2 = pd.DataFrame(data = df2, columns=["scalar_coupling_constant"])
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df2.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y2,test_size = 0.2, random_state=10)

    y3 = pd.DataFrame(data = df3, columns=["scalar_coupling_constant"])
    X_train3, X_test3, y_train3, y_test3 = train_test_split(df3.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y3,test_size = 0.2, random_state=10)

    y4 = pd.DataFrame(data = df4, columns=["scalar_coupling_constant"])
    X_train4, X_test4, y_train4, y_test4 = train_test_split(df4.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y4,test_size = 0.2, random_state=10)

    y5 = pd.DataFrame(data = df5, columns=["scalar_coupling_constant"])
    X_train5, X_test5, y_train5, y_test5 = train_test_split(df5.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y5,test_size = 0.2, random_state=10)

    y6 = pd.DataFrame(data = df6, columns=["scalar_coupling_constant"])
    X_train6, X_test6, y_train6, y_test6 = train_test_split(df6.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y6,test_size = 0.2, random_state=10)

    y7 = pd.DataFrame(data = df7, columns=["scalar_coupling_constant"])
    X_train7, X_test7, y_train7, y_test7 = train_test_split(df7.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y7,test_size = 0.2, random_state=10)

    y8 = pd.DataFrame(data = df8, columns=["scalar_coupling_constant"])
    X_train8, X_test8, y_train8, y_test8 = train_test_split(df8.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1', 'atom_0', 'atom_1', 'atom_0.1', 'atom_1.1','scalar_coupling_constant'], axis=1),y8,test_size = 0.2, random_state=10)

    y_train1 = y_train1.to_numpy()
    y_train1 = y_train1.ravel()
    y_train1 = np.array(y_train1).astype(int)
    y_test1 = y_test1.to_numpy()
    y_test1 = y_test1.ravel()
    y_test1 = np.array(y_test1).astype(int)

    y_train2 = y_train2.to_numpy()
    y_train2 = y_train2.ravel()
    y_train2 = np.array(y_train2).astype(int)
    y_test2 = y_test2.to_numpy()
    y_test2 = y_test2.ravel()
    y_test2 = np.array(y_test2).astype(int)

    y_train3 = y_train3.to_numpy()
    y_train3 = y_train3.ravel()
    y_train3 = np.array(y_train3).astype(int)
    y_test3 = y_test3.to_numpy()
    y_test3 = y_test3.ravel()
    y_test3 = np.array(y_test3).astype(int)

    y_train4 = y_train4.to_numpy()
    y_train4 = y_train4.ravel()
    y_train4 = np.array(y_train4).astype(int)
    y_test4 = y_test4.to_numpy()
    y_test4 = y_test4.ravel()
    y_test4 = np.array(y_test4).astype(int)

    y_train5 = y_train5.to_numpy()
    y_train5 = y_train5.ravel()
    y_train5 = np.array(y_train5).astype(int)
    y_test5 = y_test5.to_numpy()
    y_test5 = y_test5.ravel()
    y_test5 = np.array(y_test5).astype(int)

    y_train6 = y_train6.to_numpy()
    y_train6 = y_train6.ravel()
    y_train6 = np.array(y_train6).astype(int)
    y_test6 = y_test6.to_numpy()
    y_test6 = y_test6.ravel()
    y_test6 = np.array(y_test6).astype(int)

    y_train7 = y_train7.to_numpy()
    y_train7 = y_train7.ravel()
    y_train7 = np.array(y_train7).astype(int)
    y_test7 = y_test7.to_numpy()
    y_test7 = y_test7.ravel()
    y_test7 = np.array(y_test7).astype(int)

    y_train8 = y_train8.to_numpy()
    y_train8 = y_train8.ravel()
    y_train8 = np.array(y_train8).astype(int)
    y_test8 = y_test8.to_numpy()
    y_test8 = y_test8.ravel()
    y_test8 = np.array(y_test8).astype(int)

    print("[INFO]: data preparation complete")

    return X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8, X_test1, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8

def model_create():

    print("[INFO]: creating model...")

    LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 128,
    'min_child_samples': 79,
    'max_depth': 9,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
    }

    model = LGBMRegressor(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)

    print("[INFO]: model creation complete")

    return model

def grid_search(X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8,
        X_test1, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8, model):

    print("[INFO]: Preparing gird search")

    data_set = input("On which data set do you want to run the GridSearchCV? Select one from 3JHC, 2JHC, 1JHC, 3JHH, 2JHH, 3JHN, 2JHN, 1JHN...   ")

    if data_set == "1JHC":
        X_train = X_train1
        y_train = y_train1
        X_test = X_test1
        y_test = y_test1
    elif data_set == "2JHH":
        X_train = X_train2
        y_train = y_train2
        X_test = X_test2
        y_test = y_test2
    elif data_set == "1JHN":
        X_train = X_train3
        y_train = y_train3
        X_test = X_test3
        y_test = y_test3
    elif data_set == "2JHN":
        X_train = X_train4
        y_train = y_train4
        X_test = X_test4
        y_test = y_test4
    elif data_set == "2JHC":
        X_train = X_train5
        y_train = y_train5
        X_test = X_test5
        y_test = y_test5
    elif data_set == "3JHH":
        X_train = X_train6
        y_train = y_train6
        X_test = X_test6
        y_test = y_test6
    elif data_set == "3JHC":
        X_train = X_train7
        y_train = y_train7
        X_test = X_test7
        y_test = y_test7
    else:
        X_train = X_train8
        y_train = y_train8
        X_test = X_test8
        y_test = y_test8

    print("[INFO]: Running grid search...")

    learning = [0.2, 0.02, 0.002]
    n_estimators = [100,500,1500] # number of trees
    num_leaves = [50, 128, 250] # large num_leaves helps improve accuracy but might lead to over-fitting
    boosting_type = ['gbdt', 'dart']
    max_bin = [255, 510] # large max_bin helps improve accuracy but might slow down training progress
    reg_alpha = [0.1, 1]
    reg_lambda = [0.3, 0.5, 1]

    # grid search for initializer, batch size and number of epochs
    param_grid = dict(max_bin=max_bin, boosting_type=boosting_type, num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=learning, reg_alpha = reg_alpha, reg_lambda = reg_lambda)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)

    grid_result = grid.fit(X_train, y_train)

    # printresults
    print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f'mean={mean:.4}, std={stdev:.4} using {param}')

    print("[INFO]: Grid search cxomplete")

def main(debug = False):
    p = 0.01 if debug else 1
    df = []

    with timer("Importing datasets: "):
        print("Importing datasets")
        X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8,X_test1, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8 = import_data()
        gc.collect();

    with timer("Creating model: "):
        print("Model creation complete")
        model = model_create()
        gc.collect();

    with timer("Running grid search: "):
        print("Grid search complete")
        grid_search(X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8,
                X_test1, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8, model)
        gc.collect();

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
