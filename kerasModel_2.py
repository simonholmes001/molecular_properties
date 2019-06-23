import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

features = pd.DataFrame("features.csv")
X_train
y_train
X_validate
y_validate

model = tf.keras.Sequential()

model.add(layers.Dense(64, input_dim = 55, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
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
          validation_data=(X_validate, y_validate))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
