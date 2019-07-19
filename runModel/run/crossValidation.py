import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "data")

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def importing_datasets():

    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

    print("Datasets X_train {}, X_val {}, y_test {} and y_val {} loaded" .format(X_train.shape, X_val.shape, y_train.shape, y_val.shape))

    return X_train, X_val, y_train, y_val

def create_model(X_train, dropout=0.1, learning=0.1, kernel='uniform'):

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

    return model

def cross_validation(model, X_train, X_val, y_train, y_val):

    def get_model(dropout=0.1, learning=0.1, kernel='uniform'):
        return create_model(X_train, dropout=dropout, learning=learning, kernel=kernel)

    # create the sklearn model for the network
    model_init_batch_epoch_CV = KerasRegressor(build_fn=get_model, verbose=1)



    # we choose the initializers that came at the top in our previous cross-validation!!
    zero = initializers.Zeros()
    ones = initializers.Ones()
    constant = initializers.Constant(value=0)
    rand = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) # cannot use this option for the moment, need to find the correct syntax
    uniform = 'uniform'

    kernel = [zero, ones, uniform]
    batches = [1000,5000,10000]
    epochs = [10, 30]
    dropout = [0.1, 0.2, 0.5]
    learning = [0.01, 0.001, 0.0001]

    # grid search for initializer, batch size and number of epochs
    param_grid = dict(batch_size=batches, epochs=epochs, dropout=dropout, kernel=kernel, learning=learning)
    grid = GridSearchCV(estimator=model_init_batch_epoch_CV,
                        param_grid=param_grid,
                        cv=3, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    # printresults
    print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f'mean={mean:.4}, std={stdev:.4} using {param}')


def main(debug = False):
    p = 0.01 if debug else 1
    df = []

    with timer("Importing Datasets: "):
        print("Importing datasets")
        X_train, X_val, y_train, y_val = importing_datasets() #"X_train.npy", "X_val.npy", "y_train.npy", "y_val.mpy"
        gc.collect();

    with timer("Creating and testing model: "):
        print("Creating and testing model")
        model = create_model(X_train)
        gc.collect();

    with timer("Running cross validation: "):
        print("Cross validation")
        cross_validation(model, X_train, X_val, y_train, y_val)
        gc.collect();

if __name__ == "__main__":
    with timer("Full model run "):
        df = main()
