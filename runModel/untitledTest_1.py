"""
Creates a train and a test dataset of exactly the same length wioth the exception that the train dataset contains the scala coupling constant values which we are trying to predict.
Will combine the train dataset with each magnetic tensor and develop a model to predict the magnetic tensor.
Then add in the real magnetic tensro to the train dataset and use the conmbined dataset to predict the scalar coupling constant.
Steps are hence:
    - step 1: combine structures and test datasets
    - step 2: cobmine test and structures datasets
    - step 3: based on testStructures dataset, create a model that predicts each magnetic tensor - run model on testStructures dataset to predict the maqgnetic tensor
    - step 4: replace predicted magnetic tensor on the trainStructures dataset with the real data and use this to predict the scalar coupling constant values
    - step 5: use the model from step 4 to predict on the testStructures+magnetic tensor data set the scalar coupling constant values
"""

from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import numpy as np
import pandas as pd

print("[INFO] preparing datasets: train and structures...")

# takes the data, stepwise, and at eacxh time builds up predicted features before doing final predictions on scalar_coupling_constant

# concat_1 builds and predicts the magnetic_shielding_tensors

train = pd.read_csv('train.csv')
structures = pd.read_csv('structures.csv')

train_0 = train.merge(structures,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_1'])
train = train_0.merge(structures,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_0'],
                                     suffixes=('_atom_1', '_atom_0'))

train.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)

train_arranged_columns = ['id', 'molecule_name', 'type',
           'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
           'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
           'scalar_coupling_constant']

train = train[train_arranged_columns]

train["dist"] = np.sqrt((train["x_atom_1"] - train["x_atom_0"])**2 + (train["y_atom_1"] - train["y_atom_0"])**2
+ (train["z_atom_1"] - train["z_atom_0"])**2)

train["dot"] = train["x_atom_1"] * train["x_atom_0"] + train["y_atom_1"] * train["y_atom_0"]
+ train["z_atom_1"] * train["z_atom_0"]

train_arranged_columns = ['id', 'molecule_name', 'type',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1', 'dist', 'dot',
                          'scalar_coupling_constant']

train = train[train_arranged_columns]

print("[INFO] one hot encoding...")

df1 = pd.get_dummies(train["atom_atom_0"])
df2 = pd.get_dummies(train["atom_atom_1"])
df3 = pd.get_dummies(train["type"])

train = pd.concat([train, df1, df2, df3], axis=1)

train.drop(['type','atom_atom_0', 'atom_atom_1'], axis=1, inplace=True)

print("[INFO] writing to csv...")

train.to_csv("trainStructures.csv", index=False)

print("[INFO] preparing datasets: test and structures...")

test = pd.read_csv('test.csv')

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
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1', 'dist', 'dot']

test = test[test_arranged_columns]

print("[INFO] one hot encoding...")

df1 = pd.get_dummies(test["atom_atom_0"])
df2 = pd.get_dummies(test["atom_atom_1"])
df3 = pd.get_dummies(test["type"])

test = pd.concat([test, df1, df2, df3], axis=1)

test.drop(['type','atom_atom_0', 'atom_atom_1'], axis=1, inplace=True)

print("[INFO] writing to csv...")

test.to_csv("testStructures.csv", index=False)

print("[INFO] completed...")

print('\nTime elasped: ', datetime.now() - startTime)
