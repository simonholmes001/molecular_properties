import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dropout

# import activation
from keras.layers import Activation
from keras.layers import LeakyReLU

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras import losses
from sklearn.utils import shuffle

from keras import metrics

from keras import regularizers
from keras.regularizers import l2
from keras import initializers

# import optimisers
from keras import optimizers
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adamax
#from keras.optimizers import SGD
#from keras.optimizers import RMSprop
#from keras.optimizers import Adam
#from keras.optimizers import Adagrad
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import layers
from keras.layers.core import Activation
from keras.models import Model

from keras.layers.normalization import BatchNormalization

from keras.layers.core import Dense

from keras.layers import Input

from keras.models import load_model

from keras.callbacks import History
history = History()

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

    df_train = pd.read_csv(os.path.join(DATA_DIR,"df_train.csv"))
    print("Dataset loaded")

    return df_train

def group_by_type(df_train):

    print("[INFO] Grouping by type...")

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

    return df1, df2, df3, df4, df5, df6, df7, df8

    print("[INFO] Grouping completed")

def train_validate(df1, df2, df3, df4, df5, df6, df7, df8):

    print("[INFO] preprocessing...")

    id1 = pd.DataFrame(data = df1, columns=["id", "molecule_name"])
    id2 = pd.DataFrame(data = df2, columns=["id", "molecule_name"])
    id3 = pd.DataFrame(data = df3, columns=["id", "molecule_name"])
    id4 = pd.DataFrame(data = df4, columns=["id", "molecule_name"])
    id5 = pd.DataFrame(data = df5, columns=["id", "molecule_name"])
    id6 = pd.DataFrame(data = df6, columns=["id", "molecule_name"])
    id7 = pd.DataFrame(data = df7, columns=["id", "molecule_name"])
    id8 = pd.DataFrame(data = df8, columns=["id", "molecule_name"])

    y1 = pd.DataFrame(data = df1, columns=["scalar_coupling_constant"])
    y2 = pd.DataFrame(data = df2, columns=["scalar_coupling_constant"])
    y3 = pd.DataFrame(data = df3, columns=["scalar_coupling_constant"])
    y4 = pd.DataFrame(data = df4, columns=["scalar_coupling_constant"])
    y5 = pd.DataFrame(data = df5, columns=["scalar_coupling_constant"])
    y6 = pd.DataFrame(data = df6, columns=["scalar_coupling_constant"])
    y7 = pd.DataFrame(data = df7, columns=["scalar_coupling_constant"])
    y8 = pd.DataFrame(data = df8, columns=["scalar_coupling_constant"])

    X_train1, y_train1 = train_test_split(df1.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y1,test_size = 0.001)
    X_train2, y_train2 = train_test_split(df2.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y2,test_size = 0.001)
    X_train3, y_train3 = train_test_split(df3.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y3,test_size = 0.001)
    X_train4, y_train4 = train_test_split(df4.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y4,test_size = 0.001)
    X_train5, y_train5 = train_test_split(df5.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y5,test_size = 0.001)
    X_train6, y_train6 = train_test_split(df6.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y6,test_size = 0.001)
    X_train7, y_train7 = train_test_split(df7.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y7,test_size = 0.001)
    X_train8, y_train8 = train_test_split(df8.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1),y8,test_size = 0.001)

    normalization = input("Which type of normalization do you want? (standardScalar, minMax, quartile, normal with l1, normal with l2, )...   ")

    print("[INFO] Preparing normalization...")

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train1 = min_max_scaler.fit_transform(X_train1)
    X_train2 = min_max_scaler.fit_transform(X_train2)
    X_train3 = min_max_scaler.fit_transform(X_train3)
    X_train4 = min_max_scaler.fit_transform(X_train4)
    X_train5 = min_max_scaler.fit_transform(X_train5)
    X_train6 = min_max_scaler.fit_transform(X_train6)
    X_train7 = min_max_scaler.fit_transform(X_train7)
    X_train8 = min_max_scaler.fit_transform(X_train8)

    print("Datasets: Prepared")

    return X_train1, y1, X_train2, y2, X_train3, y3, X_train4, y4, X_train5, y5, X_train6, y6, X_train7, y7, X_train8, y8

def create_model(X_train1, y1, X_train2, y2, X_train3, y3, X_train4, y4, X_train5, y5, X_train6, y6, X_train7, y7, X_train8, y8):

    # L1 = [{'batch_size': 1000, 'dropout': 0.2, 'epochs': 30, 'kernel': <keras.initializers.Ones object at 0x7f2532713128>, 'learning': 0.01}] # 1JHC

    model1 = tf.keras.Sequential()
    model1.add(layers.Dense(64, input_dim=X_train1.shape[1], kernel_initializer=ones, activation='relu'))
    model1.add(layers.Dropout(0.2))
    model1.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model1.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model1.add(layers.Dropout(0.2))
    model1.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)

    model1.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model1.output_shape
    model1.summary()
    model1.get_config()
    model1.get_weights()

    print("[INFO] training model 1...")

    history = model1.fit(X_train1, y_train1,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    # we choose the initializers that came at the top in our previous cross-validation!!
    zero = initializers.Zeros()
    ones = initializers.Ones()
    constant = initializers.Constant(value=0)
    rand = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) # cannot use this option for the moment, need to find the correct syntax
    uniform = 'uniform'

    model1.save('model1.h5')

    print("[INFO] Preparing model 2...")

    # L2 = [{'batch_size': 1000, 'dropout': 0.2, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 2JHH

    model2 = tf.keras.Sequential()
    model2.add(layers.Dense(64, input_dim=X_train2.shape[1], kernel_initializer='uniform', activation='relu'))
    model2.add(layers.Dropout(0.2))
    model2.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model2.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model2.add(layers.Dropout(0.2))
    model2.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model2.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model2.output_shape
    model2.summary()
    model2.get_config()
    model2.get_weights()

    print("[INFO] training model 2...")

    history = model2.fit(X_train2, y_train2,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    model2.save('model2.h5')

    print("[INFO] Preparing model 3...")

    # L3 = [{'batch_size': 1000, 'dropout': 0.2, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 1JHN
    # L4 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 2JHN
    # L5 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 2JHC
    # L6 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 3JHH
    # L7 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 3JHC
    # L8 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 3JHN



    model3 = tf.keras.Sequential()
    model3.add(layers.Dense(64, input_dim=X_train3.shape[1], kernel_initializer='uniform', activation='relu'))
    model3.add(layers.Dropout(0.2))
    model3.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model3.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model3.add(layers.Dropout(0.2))
    model3.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model3.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model3.output_shape
    model3.summary()
    model3.get_config()
    model3.get_weights()

    print("[INFO] training model 3...")

    history = model3.fit(X_train3, y_train3,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    model3.save('model3.h5')

    print("[INFO] Preparing model 4...")

    # L4 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 2JHN
    # L5 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 2JHC
    # L6 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 3JHH
    # L7 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 3JHC
    # L8 = [{'batch_size': 1000, 'dropout': 0.1, 'epochs': 30, 'kernel': 'uniform', 'learning': 0.001}] # 3JHN

    model4 = tf.keras.Sequential()
    model4.add(layers.Dense(64, input_dim=X_train4.shape[1], kernel_initializer='uniform', activation='relu'))
    model4.add(layers.Dropout(0.1))
    model4.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model4.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model4.add(layers.Dropout(0.1))
    model4.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model4.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model4.output_shape
    model4.summary()
    model4.get_config()
    model4.get_weights()

    print("[INFO] training model 4...")

    history = model4.fit(X_train4, y_train4,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    print("[INFO] Preparing model 5...")

    model5.save('model5.h5')

    model5 = tf.keras.Sequential()
    model5.add(layers.Dense(64, input_dim=X_train5.shape[1], kernel_initializer='uniform', activation='relu'))
    model5.add(layers.Dropout(0.1))
    model5.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model5.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model5.add(layers.Dropout(0.1))
    model5.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model5.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model5.output_shape
    model5.summary()
    model5.get_config()
    model5.get_weights()

    print("[INFO] training model 5...")

    history = model5.fit(X_train5, y_train5,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    model5.save('model5.h5')

    print("[INFO] Preparing model 6...")

    model6 = tf.keras.Sequential()
    model6.add(layers.Dense(64, input_dim=X_train6.shape[1], kernel_initializer='uniform', activation='relu'))
    model6.add(layers.Dropout(0.1))
    model6.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model6.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model6.add(layers.Dropout(0.1))
    model6.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model6.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model6.output_shape
    model6.summary()
    model6.get_config()
    model6.get_weights()

    print("[INFO] training model 6...")

    history = model6.fit(X_train6, y_train6,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    model6.save('model6.h5')

    print("[INFO] Preparing model 7...")

    model7 = tf.keras.Sequential()
    model7.add(layers.Dense(64, input_dim=X_train7.shape[1], kernel_initializer='uniform', activation='relu'))
    model7.add(layers.Dropout(0.1))
    model7.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model7.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model7.add(layers.Dropout(0.1))
    model7.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model7.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model7.output_shape
    model7.summary()
    model7.get_config()
    model7.get_weights()

    print("[INFO] training model 7...")

    history = model7.fit(X_train7, y_train7,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    model7.save('model7.h5')

    print("[INFO] Preparing model 8...")

    model8 = tf.keras.Sequential()
    model8.add(layers.Dense(64, input_dim=X_train8.shape[1], kernel_initializer='uniform', activation='relu'))
    model8.add(layers.Dropout(0.1))
    model8.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model8.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model8.add(layers.Dropout(0.1))
    model8.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model4.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model8.output_shape
    model8.summary()
    model8.get_config()
    model8.get_weights()

    print("[INFO] training model 8...")

    history = model8.fit(X_train8, y_train8,
                epochs=30,
                verbose=1,
                batch_size=1000)

    # list all data in history
    print(history.history.keys())

    model8.save('model8.h5')

def main(debug = False):
    p = 0.01 if debug else 1
    df = []

    with timer("Importing datasets: "):
        print("Importing datasets")
        df_train = import_data()
        gc.collect();

    with timer("Grouping by type: "):
        print("Group by type")
        df1, df2, df3, df4, df5, df6, df7, df8 = group_by_type(df_train)
        gc.collect();

    with timer("Preparing to train: "):
        print("Training complete")
        X_train1, y1, X_train2, y2, X_train3, y3, X_train4, y4, X_train5, y5, X_train6, y6, X_train7, y7, X_train8, y8
        = train_validate(df1, df2, df3, df4, df5, df6, df7, df8)
        gc.collect();

    with timer("Creating and testing model: "):
        print("Creating and testing model")
        create_model(X_train1, y1, X_train2, y2, X_train3, y3, X_train4, y4, X_train5, y5, X_train6, y6, X_train7, y7, X_train8, y8)
        gc.collect();

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
