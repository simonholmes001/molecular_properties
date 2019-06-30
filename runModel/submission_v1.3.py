from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

# load and evaluate a saved model
from keras.models import load_model

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


print("[INFO] loading model...")

# load model
model = tf.keras.models.load_model('model_1JHN.h5')
#model = load_model('model_1.5.h5')
# summarize model.
model.summary()

print("[INFO] loading dataset and preprocessing...")

test = pd.read_csv("test.csv")
structures = pd.read_csv('structures.csv')

test_0 = test.merge(structures,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_1'])
test = test_0.merge(structures,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_0'],
                                     suffixes=('_atom_1', '_atom_0'))

test.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)

test_arranged_columns = ['id', 'molecule_name', 'type',
           'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
           'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1']

test = test[test_arranged_columns]

test["dist"] = np.sqrt((test["x_atom_1"] - test["x_atom_0"])**2 + (test["y_atom_1"] - test["y_atom_0"])**2
+ (test["z_atom_1"] - test["z_atom_0"])**2)

test["dot"] = test["x_atom_1"] * test["x_atom_0"] + test["y_atom_1"] * test["y_atom_0"]
+ test["z_atom_1"] * test["z_atom_0"]

test_arranged_columns = ['id', 'molecule_name', 'type',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1']

test = test[test_arranged_columns]

df5 = pd.DataFrame(data=test.groupby('molecule_name').size(), columns=["size"])
test = test.merge(df5, how="left", on=['molecule_name'])

df_3JHC = pd.DataFrame(data = test.loc[test['type'] == "3JHC"])
df_2JHC = pd.DataFrame(data = test.loc[test['type'] == "2JHC"])
df_1JHC = pd.DataFrame(data = test.loc[test['type'] == "1JHC"])
df_3JHH = pd.DataFrame(data = test.loc[test['type'] == "3JHH"])
df_2JHH = pd.DataFrame(data = test.loc[test['type'] == "2JHH"])
df_3JHN = pd.DataFrame(data = test.loc[test['type'] == "3JHN"])
df_2JHN = pd.DataFrame(data = test.loc[test['type'] == "2JHN"])
df_1JHN = pd.DataFrame(data = test.loc[test['type'] == "1JHN"])

test_predict = pd.DataFrame(data = df_3JHC)

id = pd.DataFrame(data = test_predict, columns=["id", "molecule_name"])

# one hot encoding

df1 = pd.get_dummies(test_predict["atom_atom_0"])
df2 = pd.get_dummies(test_predict["atom_atom_1"])

test_predict = pd.concat([test_predict, df1, df2], axis=1)

test_predict.drop(['id', 'molecule_name', 'type', 'atom_atom_0', 'atom_atom_1'], axis=1, inplace=True)

print("[INFO] running predictions...")

# make a prediction
y_predict = model.predict(test_predict)

test_Results = pd.DataFrame(data = id, columns=["id", "molecule_name"])
y_predict = pd.DataFrame(y_predict, columns=["scalar_coupling_constant"])
testResults = test_Results.join(y_predict)

print(testResults.head())

testResults.to_csv('testResults_1JHN.csv', index=False)

print("[INFO] complete...")

print('\nTime elasped: ', datetime.now() - startTime)
