# Imports.
import pandas as pd
import numpy as np

# Read Time Series Data from Year 1961 to 2021.
data = pd.read_csv('data/Glasnevin.csv', delimiter=',', parse_dates=['date'])
data = data.loc[7305:37430, ['date', 'maxt', 'mint']]
data.reset_index(drop=True, inplace=True)
print("---------------------------------------------------------------------------------------------------------------")
print("Time Series:")
print(data)
print("---------------------------------------------------------------------------------------------------------------")

# Data Preprocessing.
# Check the Dates.
data['date'] = pd.to_datetime(data['date']).dt.date
print('Start Date:')
print(data['date'].min())
print('\nEnd Date:')
print(data['date'].max())
print('\nDays Count:')
print((data['date'].max()-data['date'].min()).days)
date_check = data.date.diff()
print("---------------------------------------------------------------------------------------------------------------")
print('Row by Row Difference:')
print(date_check)
print('\nDifference Count:')
print(date_check.value_counts())

# Check Missing Values.
data['maxt'] = pd.to_numeric(data['maxt'], errors='coerce')  # Convert 'space' Character to 'NaN'.
data['mint'] = pd.to_numeric(data['mint'], errors='coerce')
print("---------------------------------------------------------------------------------------------------------------")
print('NaN Values:')
print('maxt:', data['maxt'].isnull().sum())
print('mint:', data['mint'].isnull().sum())

# Average Temperature Creation.
avgTemp = (data['maxt'].values + data['mint'].values)/2
data.insert(1, "AvgTemp", avgTemp, allow_duplicates=True)
del data['maxt']
del data['mint']
print("---------------------------------------------------------------------------------------------------------------")
print('Average Temperature Values:')
print(avgTemp)
print('\nTime Series:')
print(data)
print('\nNaN Values:')
print('AvgTemp:', data['AvgTemp'].isnull().sum())

# Drop Duplicate Dates with NaN Values.
for index in np.arange(1, len(date_check)):
    if date_check[index].days == 0:
        data.drop(index)

print("---------------------------------------------------------------------------------------------------------------")
print('Time Series:')
print(data)

# Index = data[data.MaxT == " "].index
# print(Index)
# lista = [Index(1), Index(2), Index(3)]
# print(lista)
#
# for i in lista:
#     data.loc[i, "MaxT"] = (data.loc[i-3:i-1, "MaxT"] + data.loc[i+1:i+3, "MaxT"])/6
#
# print(data[data.MaxT == " "].index)
