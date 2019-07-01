import pandas as pd

df0 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/testResults_XX_atom_0.csv")
df1 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/testResults_XY_atom_0.csv")
df2 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/testResults_YX_atom_0.csv")
df3 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/testResults_YY_atom_0.csv")
df4 = pd.read_csv("//Users/simonholmes/molecular_properties/runModel/testResults_ZX_atom_0.csv")

df10 = df0.merge(df1, on="molecule_name", how='left')
df11 = df10.merge(df2, on="molecule_name", how='left')
df12 = df11.merge(df3, on="molecule_name", how='left')
df13 = df12.merge(df4, on="molecule_name", how='left')

print(df13.head())
