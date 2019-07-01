"""
Adds the magnetic tensor to the trainStructures data set and will be used to predict the magnetic tensor on the testStructures dataset.
Steps are hence:
    - step 1: combine testStructures and magnetic tensor datasets
    - step 2: create subsets wiht only one magnetic tensor value (18 subsets in total)
    - step 3: create model for each subset to predict specifi l;agnetic tensor on testStructures data set
"""

from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import numpy as np
import pandas as pd

print("[INFO] preparing datasets: trainStructures and magnetic_shielding_tensors dataset...")

trainStructures = pd.read_csv('trainStructures.csv')
magnetic_shielding_tensors = pd.read_csv('magnetic_shielding_tensors.csv')

print("[INFO] combibne trainStructures with magnetic shielding tensors...")

trainStructures_0 = trainStructures.merge(magnetic_shielding_tensors,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_1'])
trainStructures = trainStructures_0.merge(magnetic_shielding_tensors,
                                    right_on=['molecule_name', 'atom_index'],
                                    left_on=['molecule_name', 'atom_index_0'],
                                     suffixes=('_atom_1', '_atom_0'))
trainStructures.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)

train_arranged_columns = ['id', 'molecule_name',
                          'atom_index_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',
                          'atom_index_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
                          'dist', 'dot', 'H', 'C', 'H.1', 'N', '1JHC', '1JHN','2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN',
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'scalar_coupling_constant']

trainStructuresMagnetic = trainStructures[train_arranged_columns]

print("[INFO] prepare datasets for each magnetic tensor to train the model...")

print("[INFO] preparing dataset for XX_atom_0...")

XX_atom_0 = trainStructuresMagnetic.drop(['YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0','XX_atom_1',
                                'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
XX_atom_0.to_csv("XX_atom_0.csv", index=False)

print("[INFO] preparing dataset for YX_atom_0...")

YX_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
YX_atom_0.to_csv("YX_atom_0.csv", index=False)

print("[INFO] preparing dataset for ZX_atom_0...")

ZX_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                    'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
ZX_atom_0.to_csv("ZX_atom_0.csv", index=False)

print("[INFO] preparing dataset for XY_atom_0...")

XY_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
XY_atom_0.to_csv("XY_atom_0.csv", index=False)

print("[INFO] preparing dataset for YY_atom_0...")

YY_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
YY_atom_0.to_csv("YY_atom_0.csv", index=False)

print("[INFO] preparing dataset for ZY_atom_0...")

ZY_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
ZY_atom_0.to_csv("ZY_atom_0.csv", index=False)

print("[INFO] preparing dataset for XZ_atom_0...")

XZ_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
XZ_atom_0.to_csv("XZ_atom_0.csv", index=False)

print("[INFO] preparing dataset for YZ_atom_0...")

YZ_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
YZ_atom_0.to_csv("YZ_atom_0.csv", index=False)

print("[INFO] preparing dataset for ZZ_atom_0...")
ZZ_atom_0 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
ZZ_atom_0.to_csv("ZZ_atom_0.csv", index=False)

print("[INFO] preparing dataset for XX_atom_1...")

XX_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
XX_atom_1.to_csv("XX_atom_1.csv", index=False)

print("[INFO] preparing dataset for YX_atom_1...")

YX_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
YX_atom_1.to_csv("YX_atom_1.csv", index=False)

print("[INFO] preparing dataset for ZX_atom_1...")

ZX_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
ZX_atom_1.to_csv("ZX_atom_1.csv", index=False)

print("[INFO] preparing dataset for XY_atom_1...")

XY_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
XY_atom_1.to_csv("XY_atom_1.csv", index=False)

print("[INFO] preparing dataset for YY_atom_1...")

YY_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
YY_atom_1.to_csv("YY_atom_1.csv", index=False)

print("[INFO] preparing dataset for ZY_atom_1...")

ZY_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
ZY_atom_1.to_csv("ZY_atom_1.csv", index=False)

print("[INFO] preparing dataset for XZ_atom_1...")

XZ_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'YZ_atom_1', 'ZZ_atom_1'], axis=1)
XZ_atom_1.to_csv("XZ_atom_1.csv", index=False)

print("[INFO] preparing dataset for YZ_atom_1...")

YZ_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'ZZ_atom_1'], axis=1)
YZ_atom_1.to_csv("YZ_atom_1.csv", index=False)

print("[INFO] preparing dataset for ZZ_atom_1...")

ZZ_atom_1 = trainStructuresMagnetic.drop(['XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                                        'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1'], axis=1)
ZZ_atom_1.to_csv("ZZ_atom_1.csv", index=False)

print("[INFO] completed...")

print('\nTime elasped: ', datetime.now() - startTime)
