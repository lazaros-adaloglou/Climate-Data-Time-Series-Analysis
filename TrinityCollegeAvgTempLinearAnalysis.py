# Imports.
import pandas as pd

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
print((data.loc[:, "MaxT"] == " ").index)
print((data.loc[:, "MinT"] == " ").sum())
