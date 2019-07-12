import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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

    #global data_set, df_train, df_test

    data_set = input("Which data set do you want to preprocess, train or test?...   ")

    print("[INFO] Loading data set: '{}'. This action will take between 20 and 60 seconds...".format(data_set))

    if data_set == "train":
        df_train = pd.read_csv(os.path.join(DATA_DIR,"df_train.csv"))
        print("Dataset {} loaded".format(data_set))
    else:
        df_test = pd.read_csv(os.path.join(DATA_DIR,"df_test.csv"))
        print("Dataset {} loaded".format(data_set))

    if data_set == "train":
        print(df_train.head())
    else:
        print(df_test.head())

    if data_set == "train":
        return df_train, data_set
    else:
        return df_test, data_set

def group_by_type(df, data_set):

    #global coupling_type, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8

    coupling_type = input("Which type do you want to preprocess? (If all, type <all>, or select one from 3JHC, 2JHC, 1JHC, 3JHH, 2JHH, 3JHN, 2JHN, 1JHN?...   ")

    print("[INFO] Grouping...")

    #if data_set == "train":
    df = df.groupby(['type'])
    df = df.get_group(coupling_type)

    df.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)

    return df, coupling_type # if model fails, delete coupling_type

    print("[INFO] Grouping completed")

def train_validate(df):

    print("[INFO] preparing X_train / y_train...")

    df = df.sample(frac=0.7, replace=True)

    id = pd.DataFrame(data = df, columns=["id", "molecule_name"])

    y = pd.DataFrame(data = df, columns=["scalar_coupling_constant"])

    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(df.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),
                                                          y,
                                                          test_size = 0.20)

    normalization = input("Which type of normalization do you want? (standardScalar, minMax, quartile, normal with l1, normal with l2, )...   ")

    print("[INFO] Preparing normalization...")

    if normalization == "standardScalar":
        scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    elif normalization == "minMax":
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
    elif normalization == "quartile":
         quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
         X_train = quantile_transformer.fit_transform(X_train)
    elif normalization == "normal with l1":
         norm = 'l1'
         X_train = preprocessing.normalize(X_train, norm=norm)
    else:
        norm = 'l2'
        X_train = preprocessing.normalize(X_train, norm=norm)


    print("Datasets: Prepared")
    print("Training sets have shape {} and {}.".format(X_train.shape, y_train.shape))
    print("Validation sets have shape {} and {}.".format(X_val.shape, y_val.shape))

    print("[INFO] saving data...")

    np.save(os.path.join(DATA_DIR,'X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR,'X_val.npy'), X_val)
    np.save(os.path.join(DATA_DIR,'y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR,'y_val.npy'), y_val)

    print("[INFO] data saved as numpy arrays...")

    print("[INFO] completed...")

def main(debug = False):
    p = 0.01 if debug else 1
    df = []

    with timer("Importing datasets: "):
        print("Importing datasets")
        df, data_set = import_data()
        gc.collect();

    with timer("Grouping by type: "):
        print("Group by type")
        df = group_by_type(df, data_set)
        gc.collect();

    with timer("Preparing X_train / y_train: "):
        print("X_train / y_train")
        df = train_validate(df)
        gc.collect();

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
