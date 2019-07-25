import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

print('[INFO] Importing test results.....')

df1 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_1.csv")
df2 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_2.csv")
df3 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_3.csv")
df4 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_4.csv")
df5 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_5.csv")
df6 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_6.csv")
df7 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_7.csv")
df8 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/testResults_8.csv")

print('[INFO] Importing complete.....')

print('[INFO] Importing and sorting test data set.....')

df_test = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/run/data/df_test.csv")

df_type = df_test[['id','molecule_name','type']]

print('[INFO] Importing complete.....')

print('[INFO] Preparing submission dataset.....')

print('[INFO] For df1.....')

df1 = df1.merge(df_type, on=['id'])

labels = df1.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df1[df1['type'] == label]

df1 = dfs["1JHC"]

df1.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] For df2.....')

df2 = df2.merge(df_type, on=['id'])

labels = df2.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df2[df2['type'] == label]

df2 = dfs["2JHH"]

df2.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] For df3.....')

df3 = df3.merge(df_type, on=['id'])

labels = df3.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df3[df3['type'] == label]

df3 = dfs["1JHN"]

df3.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] For df4.....')

df4 = df4.merge(df_type, on=['id'])

labels = df4.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df4[df4['type'] == label]

df4 = dfs["2JHN"]

df4.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] For df5.....')

df5 = df5.merge(df_type, on=['id'])

labels = df5.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df5[df5['type'] == label]

df5 = dfs["2JHC"]

df5.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] For df6.....')

df6 = df6.merge(df_type, on=['id'])

labels = df6.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df6[df6['type'] == label]

df6 = dfs["3JHH"]

df6.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] For df7.....')

df7 = df7.merge(df_type, on=['id'])

labels = df7.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df7[df7['type'] == label]

df7 = dfs["3JHC"]

df7.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] For df8.....')

df8 = df8.merge(df_type, on=['id'])

labels = df8.type.unique()

dfs = {}
for label in labels:
    dfs[label] = df8[df8['type'] == label]

df8 = dfs["3JHN"]

df8.drop(['molecule_name_x', 'molecule_name_y', 'type'], axis=1, inplace=True)

print('[INFO] Preparation complete.....')

print('[INFO] Preparing submission file.....')

df_submissiom = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

print(df_submissiom.head())

df_submissiom.sort_values(by=['id'], inplace=True)

print(df_submissiom.head())

print(df_submissiom.shape)

print('[INFO] Writing submission file to csv.....')

df_submissiom.to_csv('//Users/simonholmes/molecular_properties/runModel/run/data/submission.csv', index=False)

print('[INFO] Process complete.....')
