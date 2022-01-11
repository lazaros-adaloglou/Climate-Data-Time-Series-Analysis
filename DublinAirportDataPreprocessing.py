# Imports.
import pandas as pd
import numpy as np

# Read Time Series Data from Year 1948 to 2021.
data = pd.read_csv('data/DublinAirport.csv', delimiter=',', parse_dates=['date'])
data = data.loc[:, ['date', 'maxtp', 'mintp']]
data.reset_index(drop=True, inplace=True)
print("---------------------------------------------------------------------------------------------------------------")
print("Original Time Series:")
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
data['maxtp'] = pd.to_numeric(data['maxtp'], errors='coerce')  # Convert 'space' Character to 'NaN'.
data['mintp'] = pd.to_numeric(data['mintp'], errors='coerce')
print("---------------------------------------------------------------------------------------------------------------")
print('NaN Values:')
print('maxtp:', data['maxtp'].isnull().sum())
print('mintp:', data['mintp'].isnull().sum())

# Average Temperature Creation.
avgTemp = (data['maxtp'].values + data['mintp'].values)/2
data.insert(1, "AvgTemp", avgTemp, allow_duplicates=True)
del data['maxtp']
del data['mintp']
print("---------------------------------------------------------------------------------------------------------------")
print('Average Temperature Values:')
print(avgTemp)
print('\nTime Series:')
print(data)
print('\nNaN Values:')
print('AvgTemp:', data['AvgTemp'].isnull().sum())

# # Drop Duplicate Dates.
# for index in np.arange(1, len(date_check)):
#     if date_check[index].days == 0:
#         data = data.drop(index)
#
# data.reset_index(drop=True, inplace=True)
# print("---------------------------------------------------------------------------------------------------------------")
# print('Time Series After Dropping Duplicate Dates:')
# print(data)
# print('\nNaN Values:', data['AvgTemp'].isnull().sum())
#
# Fill Missing Dates with NaN.
# complete_date = pd.date_range(min(data.date), max(data.date)).date
# new_date = []
# indexes = []
# for i, item in enumerate(complete_date):
#     if item not in data.date.values:
#         new_date.append(item)
#         indexes.append(i)
#
# new_data = pd.DataFrame({'date': new_date, 'AvgTemp': [np.nan]*len(new_date)}, index=indexes)
# month1 = pd.DataFrame({'date': new_data.date[0:31], 'AvgTemp': new_data.AvgTemp[0:31]}, index=indexes[0:31])
# month2 = pd.DataFrame({'date': new_data.date[31:62], 'AvgTemp': new_data.AvgTemp[31:62]}, index=indexes[31:62])
# data = pd.concat([data.iloc[0:indexes[0]], month1, data.iloc[indexes[0]:indexes[31]-31],
#                   month2, data.iloc[indexes[31]-31:]]).reset_index(drop=True)
# print("---------------------------------------------------------------------------------------------------------------")
# print('Time Series After Filling Missing Date Temperature with NaN:')
# print(data)
# print('\nNaN Values:', data['AvgTemp'].isnull().sum())
#
# # Fill Missing Values.
# avgVal = [0]*366
# days = np.arange(1, 366)
# years = np.arange(0, 60)
# for day in days:
#     counter = 0
#     for year in years:
#         val = data.loc[365 * year + day - 1, 'AvgTemp']
#         if not pd.isnull(val):
#             avgVal[day] = avgVal[day] + val
#             counter = counter + 1
#     avgVal[day] = avgVal[day] / counter
#
# avgData = pd.DataFrame({'AvgVal': avgVal})
#
# for i in data.index.values:
#     if pd.isnull(data.loc[i, 'AvgTemp']):
#         data.loc[i, 'AvgTemp'] = avgData.loc[i % 365, 'AvgVal']
#
print("---------------------------------------------------------------------------------------------------------------")
# print('\nNaN Values After Filling Them:', data['AvgTemp'].isnull().sum())
# date_check = data.date.diff()
# print('\nDate Steps Count:\n')
# print(date_check.value_counts())
filename = 'data/DublinAirport_Data.csv'
print('\nData Exported to', filename, '\n')
data.to_csv(filename)
