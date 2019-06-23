"""
Script to merge all data into one data frame
Credits to https://www.kaggle.com/kernels/scriptcontent/15799375/download
who did this with much more intelligence than me
"""

from datetime import datetime
startTime = datetime.now()

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#import os
#print(os.listdir("../"))

# import the datasets

train_dataset = pd.read_csv('train.csv')
scalar_coupling_contributions = pd.read_csv('scalar_coupling_contributions.csv')
structures = pd.read_csv('structures.csv')
magnetic_shielding_tensors = pd.read_csv('magnetic_shielding_tensors.csv')
mulliken_charges = pd.read_csv('mulliken_charges.csv')
dipole_moments = pd.read_csv('dipole_moments.csv')
potential_energy = pd.read_csv('potential_energy.csv')

# combine the datasets

train_dataset = pd.concat([train_dataset, scalar_coupling_contributions[['fc', 'sd', 'pso', 'dso']]], axis=1)

train_dataset_0 = train_dataset.merge(structures,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_1'])
train_dataset = train_dataset_0.merge(structures,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_0'],
                                     suffixes=('_atom_1', '_atom_0'))

train_dataset.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)

train_arranged_columns = ['id', 'molecule_name', 'type',
           'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
           'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
           'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']

train_dataset = train_dataset[train_arranged_columns]

train_dataset_0 = train_dataset.merge(magnetic_shielding_tensors,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_1'])
train_dataset = train_dataset_0.merge(magnetic_shielding_tensors,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_0'],
                                     suffixes=('_atom_1', '_atom_0'))
train_dataset.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)

train_arranged_columns = ['id', 'molecule_name', 'type',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']

train_dataset = train_dataset[train_arranged_columns]

train_dataset_0 = train_dataset.merge(mulliken_charges,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_1'])
train_dataset = train_dataset_0.merge(mulliken_charges,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_0'],
                                     suffixes=('_atom_1', '_atom_0'))
train_dataset.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)

train_arranged_columns = ['id', 'molecule_name', 'type',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']

train_dataset = train_dataset[train_arranged_columns]

train_dataset = train_dataset.merge(dipole_moments,
                                    right_on='molecule_name',
                                    left_on='molecule_name')

train_arranged_columns = ['id', 'molecule_name', 'type', 'X', 'Y', 'Z',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']

train_dataset = train_dataset[train_arranged_columns]

train_dataset = train_dataset.merge(potential_energy,
                                    right_on='molecule_name',
                                    left_on='molecule_name')

train_arranged_columns = ['id', 'molecule_name', 'type', 'X', 'Y', 'Z', 'potential_energy',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']

train_dataset = train_dataset[train_arranged_columns]

train_dataset["dist"] = np.sqrt((train_dataset["x_atom_1"] - train_dataset["x_atom_0"])**2 + (train_dataset["y_atom_1"] - train_dataset["y_atom_0"])**2
+ (train_dataset["z_atom_1"] - train_dataset["z_atom_0"])**2)

train_dataset["dot"] = train_dataset["x_atom_1"] * train_dataset["x_atom_0"] + train_dataset["y_atom_1"] * train_dataset["y_atom_0"]
+ train_dataset["z_atom_1"] * train_dataset["z_atom_0"]

train_arranged_columns = ['id', 'molecule_name', 'type', 'X', 'Y', 'Z', 'potential_energy',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'dist','dot', 'scalar_coupling_constant']

train_dataset = train_dataset[train_arranged_columns]

# one hot encoding

df_2 = pd.get_dummies(train_dataset["type"])
df_3 = pd.get_dummies(train_dataset["atom_atom_0"])
df_4 = pd.get_dummies(train_dataset["atom_atom_1"])

train_dataset = pd.concat([train_dataset, df_2, df_3, df_4], axis=1)

df5 = pd.DataFrame(data=train_dataset.groupby('molecule_name').size(), columns=["size"])
train_dataset = train_dataset.merge(df5, how="left", on=['molecule_name'])

train_dataset.drop(['molecule_name', 'type',  'atom_atom_0', 'atom_atom_1'], axis=1, inplace=True)

train_arranged_columns = ['id','size', '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN', 'H', 'C', 'H', 'N', 'X', 'Y', 'Z', 'potential_energy',
                          'atom_index_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'dist','dot', 'scalar_coupling_constant']

train_dataset = train_dataset[train_arranged_columns]

train_dataset.to_csv('features.csv', index=False)

print('\nTime elasped: ', datetime.now() - startTime)
