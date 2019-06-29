print("[INFO] loading dependencies...")

# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

import tensorflow as tf
import numpy as np
import pandas as pd

print("[INFO] loading model...")

# load model
model = tf.keras.models.load_model('model_1.5.h5')
#model = load_model('model_1.5.h5')
# summarize model.
model.summary()

print("[INFO] loading dataset...")

# load dataset
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("[INFO] predictions on test dataset...")

# make a prediction
y_predict = model.predict(X_test)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (y_test, y_predict)) # (y_test[0], y_predict[0]))

predict = []
test = []

for i in y_predict:
    predict.append(i)
for i in y_test:
    test.append(i)

test_score = pd.DataFrame({"prediction":predict, "test":test})
test_score["result"] = test_score["prediction"] - test_score["test"]

test_score.head()

test_score.to_csv('testScore.csv', index=False)

print("[INFO] complete...")
