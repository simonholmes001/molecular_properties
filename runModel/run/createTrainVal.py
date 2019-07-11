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

    global data_set, df_train, df_test

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

def group_by_type():

    global coupling_type, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8

    coupling_type = input("Which type do you want to preprocess? (If all, type <all>, or select one from 3JHC, 2JHC, 1JHC, 3JHH, 2JHH, 3JHN, 2JHN, 1JHN?...   ")

    print("[INFO] Grouping...")

    if data_set == "train":
        df = df_train.groupby(['type'])
        df_1 = df.get_group('3JHC')
        df_2 = df.get_group('2JHC')
        df_3 = df.get_group('1JHC')
        df_4 = df.get_group('3JHH')
        df_5 = df.get_group('2JHH')
        df_6 = df.get_group('3JHN')
        df_7 = df.get_group('2JHN')
        df_8 = df.get_group('1JHN')
    else:
        df = df_test.groupby(['type'])
        df_1 = df.get_group('3JHC')
        df_2 = df.get_group('2JHC')
        df_3 = df.get_group('1JHC')
        df_4 = df.get_group('3JHH')
        df_5 = df.get_group('2JHH')
        df_6 = df.get_group('3JHN')
        df_7 = df.get_group('2JHN')
        df_8 = df.get_group('1JHN')

    df_1.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df_2.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df_3.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df_4.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df_5.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df_6.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df_7.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df_8.drop(['type', 'atom_0', 'atom_1'], axis=1, inplace=True)

    if coupling_type == "3JHC":
        return df_1
    elif coupling_type == "2JHC":
        return df_2
    elif coupling_type == "1JHC":
        return df_3
    elif coupling_type == "3JHH":
        return df_4
    elif coupling_type == "2JHH":
        return df_5
    elif coupling_type == "3JHN":
        return df_6
    elif coupling_type == "2JHN":
        return df_7
    else:
        return df_8

    print("[INFO] Grouping completed")

def train_validate():

    if coupling_type == "3JHC":
        df = df_1
    elif coupling_type == "2JHC":
        df = df_2
    elif coupling_type == "1JHC":
        df = df_3
    elif coupling_type == "3JHH":
        df =  df_4
    elif coupling_type == "2JHH":
        df =  df_5
    elif coupling_type == "3JHN":
        df =  df_6
    elif coupling_type == "2JHN":
        df =  df_7
    else:
        df =  df_8

    print("[INFO] preparing X_train / y_train...")

    df = df.sample(frac=0.7, replace=True)

    id = pd.DataFrame(data = df, columns=["id", "molecule_name"])

    y = pd.DataFrame(data = df, columns=["scalar_coupling_constant"])

    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(df.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),
                                                          y,
                                                          test_size = 0.20)

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
        df = import_data()
        gc.collect();

    with timer("Grouping by type: "):
        print("Group by type")
        df = group_by_type()
        gc.collect();

    with timer("Preparing X_train / y_train: "):
        print("X_train / y_train")
        df = train_validate()
        gc.collect();

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
