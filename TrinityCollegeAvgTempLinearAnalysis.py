# Imports.
import pandas as pd
import numpy as np

# Read Time Series Data from Year 1961 to 2021.
df = pd.read_csv("data/Glasnevin.csv", delimiter=',')
df = df.loc[7305:37430, ['date', 'maxt', 'mint']]

print(df)
print((df.loc[:, "maxt"] == " ").sum())
print((df.loc[:, "mint"] == " ").sum())

df = df.drop(df[(df.maxt == ' ') & (df.mint == ' ')].index)
print(df)

maxTemp = df.loc[:, "maxt"].values
minTemp = df.loc[:, "mint"].values
print(maxTemp)
print(minTemp)

data = pd.DataFrame({"MaxT": maxTemp, "MinT": minTemp})
print(data)
print(data[data.MaxT == " "].index)
print(data[data.MinT == " "].index)

Index = data[data.MaxT == " "].index.values
print(Index)
lista = [Index(1), Index(2), Index(3)]
print(lista)

for i in lista:
    data.loc[i, "MaxT"] = (data.loc[i-3:i-1, "MaxT"] + data.loc[i+1:i+3, "MaxT"])/6

print(data[data.MaxT == " "].index)
