from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import Dropout

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import SGD

from keras import losses
from keras import metrics
from keras import regularizers
from keras import initializers
from keras import optimizers

from keras.models import load_model

from keras.callbacks import History
history = History()

print("[INFO] loading dataset...")

X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
#X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
#y_test = np.load('y_test.npy')

print("[INFO] build and compiling model...")

zero = initializers.Zeros()
kernel_initializer=zero
kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)

model = tf.keras.Sequential()
model.add(layers.Dense(20, activation='relu', input_dim=X_train.shape[1], kernel_initializer=zero,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
#model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None))
model.add(layers.Dense(20, activation='relu', kernel_initializer=zero,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='relu', kernel_initializer=zero,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

rms = RMSprop(lr=0.00001)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#ada = Adagrad(lr=0.01, epsilon=None, decay=0.0)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

mse = 'mse'
#log = losses.mean_squared_logarithmic_error(y_train, y_val)

model.compile(optimizer=rms, # 'rmsprop
                    loss=mse,
                    metrics=['mae'])

model.output_shape
model.summary()
model.get_config()
model.get_weights()

print("[INFO] training model...")

history = model.fit(X_train, y_train,
            epochs=1000, #200 for images 1.2, 2.2 and model_1.2 / 300 for images 1.3, 2.3 and model_1.3 (based on small_features_3)
            verbose=1,
            batch_size=10000,
            validation_data=(X_val, y_val))

print("[INFO] evaluating model...")

score = model.evaluate(X_test,
                        y_test
                        )

print(score)

# list all data in history
print(history.history.keys())

# summarise history for accuracy

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("1.5.png")
plt.clf()

# summarise history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("2.5.png")

#prediction = model.predict(X_test, batch_size=50)

"""
SAVE MODEL WITH SAME TAG AS MAGNETIC SHIELDING TENSOR
"""
model.save('model_trainStructures.h5')

# show the inputs and predicted outputs
#for i in range(len(X_test)):
#	print("X=%s, Predicted=%s" % (X_test[i], y_test[i]))

print('\nTime elasped: ', datetime.now() - startTime)
