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

def train_validate(df):

    print("[INFO] preparing X_train / y_train...")

    y = pd.DataFrame(data = df, columns=["scalar_coupling_constant"])

    # Split the 'features' and 'income' data into training and testing sets
    X_train = df.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)

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

    print("[INFO] saving data...")

    print("[INFO] completed...")

    return X_train, y_train

def create_model(X_train, dropout=0.1, learning=0.1, kernel='uniform', coupling_type): # MUST FIX EACH OF THESE PARAMETRES BEASED ON CORSS VALIDATION

    model = tf.keras.Sequential()
    model.add(layers.Dense(64, input_dim=X_train.shape[1], kernel_initializer=kernel, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(layers.Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    # compile model

    rms = RMSprop(lr=learning, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='mse',
                  optimizer=rms,
                  metrics=['mae'])

    model.output_shape
    model.summary()
    model.get_config()
    model.get_weights()

    print("[INFO] training model...")

    history = model.fit(X_train, y_train,
                epochs=30,
                verbose=1,
                batch_size=10000)

    # list all data in history
    print(history.history.keys())

    # summarise history for accuracy

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("test.png")
    plt.clf()

    # we choose the initializers that came at the top in our previous cross-validation!!
    zero = initializers.Zeros()
    ones = initializers.Ones()
    constant = initializers.Constant(value=0)
    rand = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) # cannot use this option for the moment, need to find the correct syntax
    uniform = 'uniform'

    kernel = [zero, ones, constant, uniform]
    batches = [1000,10000]
    epochs = [10, 20, 30]
    dropout = [0.1, 0.2, 0.5]
    learning = [0.1, 0.01, 0.001, 0.0001]

    model.save('model {}.h5'.format(coupling_type))

def main(debug = False):
    p = 0.01 if debug else 1
    df = []

    with timer("Importing datasets: "):
        print("Importing datasets")
        df, data_set = import_data()
        gc.collect();

    with timer("Grouping by type: "):
        print("Group by type")
        df, coupling_type = group_by_type(df, data_set)
        gc.collect();

    with timer("Preparing X_train / y_train: "):
        print("X_train / y_train")
        X_train, y_train = train_validate(df)
        gc.collect();

    with timer("Creating and testing model: "):
        print("Creating and testing model")
        model = create_model(X_train, coupling_type)
        gc.collect();

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
