import pandas as pd
import numpy as np

df = pd.read_csv("data/TrinityCollege.csv", delimiter=',')
df = df.loc[7305:12734, ['date', 'maxt', 'mint']]
# print(df)
# print((df.loc[:, "maxt"] == " ").sum())
# print((df.loc[:, "mint"] == " ").sum())

for space in np.arange(7305, 12734+1):
    if df.loc[space, "maxt"] == " " and df.loc[space, "mint"] == " ":
        df.drop([space], axis=0)

print(df)
