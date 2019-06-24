from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import pandas as pd

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
from keras import regularizers

from keras.models import load_model

from keras.callbacks import History
history = History()

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

print("[INFO] loading dataset and preprocessing the data...")

features = pd.read_csv("small_features.csv")
X = features.drop(["fc", "sd", "pso", "dso", "scalar_coupling_constant"], axis=1)
y = pd.DataFrame(data = features, columns=["scalar_coupling_constant"])

X = preprocessing.normalize(X, norm='l2')

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.2)

X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# build the model

print("[INFO] build and compiling model...")

model = tf.keras.Sequential()

model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_dim=X_train.shape[1])) # how to work in Dropout???
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])

model.output_shape
model.summary()
model.get_config()
model.get_weights()

print("[INFO] training model...")

history = model.fit(X_train, y_train,
            epochs=50,
            verbose=1,
            batch_size=50,
            validation_data=(X_val, y_val))

print("[INFO] evaluating model...")

score = model.evaluate(X_test,
                        y_test,
                        batch_size=50)

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
plt.savefig("1.2.png")

# summarise history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("2.2.png")

#prediction = model.predict(X_test, batch_size=50)

model.save('model_1.2.h5')

# show the inputs and predicted outputs
#for i in range(len(X_test)):
#	print("X=%s, Predicted=%s" % (X_test[i], y_test[i]))

print('\nTime elasped: ', datetime.now() - startTime)
