from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import pandas as pd
import numpy as np

from tempfile import TemporaryFile

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

print("[INFO] loading datasets...")

train_df = pd.read_csv("train_df.csv")

#df1 = pd.get_dummies(Z["atom_0"])
#df2 = pd.get_dummies(Z["atom_1"])

#Z = pd.concat([Z, df1, df2], axis=1)

#Z.drop(['atom_0', 'atom_1'], axis=1, inplace=True)

print("[INFO] preparing datasets...")

train_df = train_df.sample(frac=0.7, replace=True)

#X.to_csv("X.csv", index=False)

# id = pd.DataFrame(data = X, columns=["id", "molecule_name"])

#y = pd.DataFrame(data = X, columns=["scalar_coupling_constant"])

#X.drop(['id', 'molecule_name','scalar_coupling_constant'], axis=1, inplace=True)

# Split the 'features' and 'income' data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(train_df.drop('scalar_coupling_constant', axis=1),
                                                      train_df['scalar_coupling_constant'],
                                                      test_size = 0.10, random_state = 21)

X_train = X_train.drop(['id', 'atom_0', 'type', 'atom_1','molecule_name'], axis=1).values
y_train = y_train.values
X_val = X_val.drop(['id', 'atom_0', 'type', 'atom_1','molecule_name'], axis=1).values
y_val = y_val.values

print("Datasets: Prepared.")
print("Training set has {} shape.".format(X_train.shape))
print("Validation set has {} shape.".format(X_val.shape))

print("[INFO] preprocessing...")

#norm = 'l2'
#test_size_val = 0.2 # 20% of samples for validation_test
#test_size_test = 0.5 # 50% of validation_test set for validation, 50% for test

#X = preprocessing.normalize(X, norm=norm)

#X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_size_val)

#X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size_test)

print(X_train.shape)
print(X_val.shape)
#print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
#print(y_test.shape)

print("[INFO] saving data...")

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
#np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
#np.save('y_test.npy', y_test)

print("[INFO] data saved as numpy arrays...")

print("[INFO] completed...")

print('\nTime elasped: ', datetime.now() - startTime)
