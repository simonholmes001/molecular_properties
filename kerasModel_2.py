from datetime import datetime
startTime = datetime.now()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from sklearn.model_selection import train_test_split

features = pd.read_csv("small_features.csv")
X = features.drop(["fc", "sd", "pso", "dso", "scalar_coupling_constant"], axis=1)
y = pd.DataFrame(data = features, columns=["id","scalar_coupling_constant"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = tf.keras.Sequential()

model.add(layers.Dense(64, input_dim = 50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
            kernel_initializer='orthogonal', bias_initializer=tf.keras.initializers.constant(2.0)))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
            kernel_initializer='orthogonal', bias_initializer=tf.keras.initializers.constant(2.0)))
model.add(layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
            kernel_initializer='orthogonal', bias_initializer=tf.keras.initializers.constant(2.0)))

model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('\nTime elasped: ', datetime.now() - startTime)
