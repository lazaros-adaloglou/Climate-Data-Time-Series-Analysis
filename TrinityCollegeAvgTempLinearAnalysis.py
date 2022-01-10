import pandas as pd
import os

df = pd.read_csv("data/TrinityCollege.csv", delimiter=',')
df = df.loc[:, ["maxt", "mint"]]
print(df.isnull().sum())

