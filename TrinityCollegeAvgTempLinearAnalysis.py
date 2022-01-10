import pandas as pd
import os

df = pd.read_csv("data/TrinityCollege.csv", delimiter=',')
df = df.loc[7305:, ['date', 'maxt', 'mint']]
#print(df)
#print((df.loc[:, "maxt"] == " ").sum())
#print((df.loc[:, "mint"] == " ").sum())
print(df.duplicated('date', keep='first'))


