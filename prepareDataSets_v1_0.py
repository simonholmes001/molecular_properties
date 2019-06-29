from datetime import datetime
startTime = datetime.now()

print("[INFO] loading dependencies...")

import pandas as pd

print("[INFO] loading dataset...")

features = pd.read_csv("features.csv")
features.sample(frac=0.5, replace=True) # generates random 50% cut of initial dataset

X = features.drop(["id", "fc", "sd", "pso", "dso", "scalar_coupling_constant"], axis=1)
y = pd.DataFrame(data = features, columns=["scalar_coupling_constant"])

X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)

print("[INFO] complete...")

print('\nTime elasped: ', datetime.now() - startTime)
