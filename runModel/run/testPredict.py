import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model

import tensorflow as tf

import os

from contextlib import contextmanager
import time
import gc

from createTrainVal import import_data, group_by_type

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "data")

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def import_data():

    print("[INFO] Loading train data set. This action will take between 20 and 60 seconds...")

    df_test = pd.read_csv(os.path.join(DATA_DIR,"df_test.csv"))
    print("Dataset loaded")

    return df_test

def group_by_type(df_test):

    print("[INFO] Grouping by type...")

    labels = df_test.type.unique()

    dfs = {}
    for label in labels:
        dfs[label] = df_test[df_test['type'] == label]

    df1 = dfs["1JHC"]
    df2 = dfs["2JHH"]
    df3 = dfs["1JHN"]
    df4 = dfs["2JHN"]
    df5 = dfs["2JHC"]
    df6 = dfs["3JHH"]
    df7 = dfs["3JHC"]
    df8 = dfs["3JHN"]

    id1 = pd.DataFrame(data = df1, columns=["id", "molecule_name"])
    id2 = pd.DataFrame(data = df2, columns=["id", "molecule_name"])
    id3 = pd.DataFrame(data = df3, columns=["id", "molecule_name"])
    id4 = pd.DataFrame(data = df4, columns=["id", "molecule_name"])
    id5 = pd.DataFrame(data = df5, columns=["id", "molecule_name"])
    id6 = pd.DataFrame(data = df6, columns=["id", "molecule_name"])
    id7 = pd.DataFrame(data = df7, columns=["id", "molecule_name"])
    id8 = pd.DataFrame(data = df8, columns=["id", "molecule_name"])

    df1.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df2.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df3.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df4.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df5.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df6.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df7.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)
    df8.drop(['id', 'molecule_name', 'type', 'atom_0', 'atom_1'], axis=1, inplace=True)

    return df1, df2, df3, df4, df5, df6, df7, df8, id1, id2, id3, id4, id5, id6, id7, id8

    print("[INFO] Grouping completed")

def predictions(df1, df2, df3, df4, df5, df6, df7, df8, id1, id2, id3, id4, id5, id6, id7, id8):

    print("[INFO] running predictions...")

    # load model
    model1 = tf.keras.models.load_model('model1.h5')
    model1.summary()
    model2 = tf.keras.models.load_model('model2.h5')
    model2.summary()
    model3 = tf.keras.models.load_model('model3.h5')
    model3.summary()
    model4 = tf.keras.models.load_model('model4.h5')
    model4.summary()
    model5 = tf.keras.models.load_model('model5.h5')
    model5.summary()
    model6 = tf.keras.models.load_model('model6.h5')
    model6.summary()
    model7 = tf.keras.models.load_model('model7.h5')
    model7.summary()
    model8 = tf.keras.models.load_model('model8.h5')
    model8.summary()

    # make a prediction
    y1 = model1.predict(df1)
    test_Results_1 = pd.DataFrame(data = id1, columns=["id", "molecule_name"])
    y_predict_1 = pd.DataFrame(y1, columns=["scalar_coupling_constant"])
    testResults_1 = test_Results_1.join(y_predict_1)

    y2 = model2.predict(df2)
    test_Results_2 = pd.DataFrame(data = id2, columns=["id", "molecule_name"])
    y_predict_2 = pd.DataFrame(y2, columns=["scalar_coupling_constant"])
    testResults_2 = test_Results_2.join(y_predict_2)

    y3 = model3.predict(df3)
    test_Results_3 = pd.DataFrame(data = id3, columns=["id", "molecule_name"])
    y_predict_3 = pd.DataFrame(y3, columns=["scalar_coupling_constant"])
    testResults_3 = test_Results_3.join(y_predict_3)

    y4 = model4.predict(df4)
    test_Results_4 = pd.DataFrame(data = id4, columns=["id", "molecule_name"])
    y_predict_4 = pd.DataFrame(y4, columns=["scalar_coupling_constant"])
    testResults_4 = test_Results_4.join(y_predict_4)

    y5 = model5.predict(df5)
    test_Results_5 = pd.DataFrame(data = id5, columns=["id", "molecule_name"])
    y_predict_5 = pd.DataFrame(y5, columns=["scalar_coupling_constant"])
    testResults_5 = test_Results_5.join(y_predict_5)

    y6 = model6.predict(df6)
    test_Results_6 = pd.DataFrame(data = id6, columns=["id", "molecule_name"])
    y_predict_6 = pd.DataFrame(y6, columns=["scalar_coupling_constant"])
    testResults_6 = test_Results_6.join(y_predict_6)

    y7 = model7.predict(df7)
    test_Results_7 = pd.DataFrame(data = id7, columns=["id", "molecule_name"])
    y_predict_7 = pd.DataFrame(y7, columns=["scalar_coupling_constant"])
    testResults_7 = test_Results_7.join(y_predict_7)

    y8 = model8.predict(df8)
    test_Results_8 = pd.DataFrame(data = id8, columns=["id", "molecule_name"])
    y_predict_8 = pd.DataFrame(y8, columns=["scalar_coupling_constant"])
    testResults_8 = test_Results_8.join(y_predict_8)

    print("[INFO] Saving predictions to csv...")

    testResults_1.to_csv('testResults_1.csv', index=False)
    testResults_2.to_csv('testResults_2.csv', index=False)
    testResults_3.to_csv('testResults_3.csv', index=False)
    testResults_4.to_csv('testResults_4.csv', index=False)
    testResults_5.to_csv('testResults_5.csv', index=False)
    testResults_6.to_csv('testResults_6.csv', index=False)
    testResults_7.to_csv('testResults_7.csv', index=False)
    testResults_8.to_csv('testResults_8.csv', index=False)

    print(testResults_1.head())
    print(testResults_2.head())
    print(testResults_3.head())
    print(testResults_4.head())
    print(testResults_5.head())
    print(testResults_6.head())
    print(testResults_7.head())
    print(testResults_8.head())

    print("[INFO] Predictions completed...")

def main(debug = False):
    p = 0.01 if debug else 1
    df = []

    with timer("Importing datasets: "):
        print("Importing datasets")
        df_test = import_data()
        gc.collect();

    with timer("Grouping by type: "):
        print("Group by type")
        df1, df2, df3, df4, df5, df6, df7, df8, id1, id2, id3, id4, id5, id6, id7, id8 = group_by_type(df_test)
        gc.collect();

    with timer("Preparing to predict: "):
        print("Predictions complete")
        predictions(ddf1, df2, df3, df4, df5, df6, df7, df8, id1, id2, id3, id4, id5, id6, id7, id8)
        gc.collect();

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
