"""
Prepare each dataset for the model to train for each magnetic tensor
"""

from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import pandas as pd
import numpy as np

from tempfile import TemporaryFile

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

print("[INFO] loading datasets...")

X = pd.read_csv("XX_atom_0.csv")

print("[INFO] preparing datasets...")

X = X.sample(frac=0.5, replace=True)

# id = pd.DataFrame(data = X, columns=["id", "molecule_name"])
"""
THIS IS WHERE WE LOAD THE MAGNETIC SHIELDING TENSOR FOR MDOEL DEVELOPMENT
"""
y = pd.DataFrame(data = X, columns=["XX_atom_0"])

X.drop(['id', 'molecule_name','scalar_coupling_constant', 'XX_atom_0'], axis=1, inplace=True)

print("[INFO] preprocessing...")

norm = 'l2'
test_size_val = 0.2 # 20% of samples for validation_test
test_size_test = 0.5 # 50% of validation_test set for validation, 50% for test

X = preprocessing.normalize(X, norm=norm)

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_size_val)

X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size_test)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

print("[INFO] saving data...")

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

print("[INFO] data saved as numpy arrays...")

print("[INFO] completed...")

print('\nTime elasped: ', datetime.now() - startTime)
