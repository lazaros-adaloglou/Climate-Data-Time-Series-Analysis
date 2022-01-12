# Imports.
import pandas as pd

# Read Time Series Data from Year 1942 to 2021.
data = pd.read_csv('data/DublinAirport.csv', delimiter=',', parse_dates=['date'])
data = data.loc[:, ['date', 'maxtp', 'mintp']]
data.reset_index(drop=True, inplace=True)
print("---------------------------------------------------------------------------------------------------------------")
print("Original Time Series:\n")
print(data)
print("---------------------------------------------------------------------------------------------------------------")

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
print('Average Temperature Time Series:\n')
print(data)
print('\nNaN Values:')
print(data['AvgTemp'].isnull().sum())

# Export Preprocessed Time Series.
print("---------------------------------------------------------------------------------------------------------------")
filename = 'data/DublinAirport_Data.csv'
print('\nData Exported to', filename, '\n')
data.to_csv(filename)
