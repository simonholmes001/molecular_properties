from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import numpy as np
import pandas as pd

print("[INFO] preparing datasets...")

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
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'scalar_coupling_constant']

train = train[train_arranged_columns]

df5 = pd.DataFrame(data=train.groupby('molecule_name').size(), columns=["size"])
train = train.merge(df5, how="left", on=['molecule_name'])

print("[INFO] preparing type datasets...")

df_3JHC = pd.DataFrame(data = train.loc[train['type'] == "3JHC"])
df_2JHC = pd.DataFrame(data = train.loc[train['type'] == "2JHC"])
df_1JHC = pd.DataFrame(data = train.loc[train['type'] == "1JHC"])
df_3JHH = pd.DataFrame(data = train.loc[train['type'] == "3JHH"])
df_2JHH = pd.DataFrame(data = train.loc[train['type'] == "2JHH"])
df_3JHN = pd.DataFrame(data = train.loc[train['type'] == "3JHN"])
df_2JHN = pd.DataFrame(data = train.loc[train['type'] == "2JHN"])
df_1JHN = pd.DataFrame(data = train.loc[train['type'] == "1JHN"])

print("[INFO] writing to csv...")

df_3JHC.to_csv('3JHC.csv', index=False)
df_2JHC.to_csv('2JHC.csv', index=False)
df_1JHC.to_csv('1JHC.csv', index=False)
df_3JHH.to_csv('3JHH.csv', index=False)
df_2JHH.to_csv('2JHH.csv', index=False)
df_3JHN.to_csv('3JHN.csv', index=False)
df_2JHN.to_csv('2JHN.csv', index=False)
df_1JHN.to_csv('1JHN.csv', index=False)

print("[INFO] completed...")

print('\nTime elasped: ', datetime.now() - startTime)
