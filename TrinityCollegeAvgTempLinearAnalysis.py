import pandas as pd
import os

df = pd.read_csv("data/TrinityCollege.csv", delimiter=',')
df = df.loc[7305:, ["maxt", "mint"]]
print((df.loc[:, ['maxt']].values == ' ').sum())