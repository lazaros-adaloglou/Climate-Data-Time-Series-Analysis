# Imports.
import pandas as pd

# Read Time Series Data from Year  to .
data = pd.read_csv('data/VIXClose.csv', delimiter=',', parse_dates=['Date'])
data = data.loc[:, ['Date', 'VIX Close']]
data.rename(columns={'VIX Close': 'VIXClose'}, inplace=True)
data.reset_index(drop=True, inplace=True)
print("---------------------------------------------------------------------------------------------------------------")
print("Original Time Series:\n")
print(data)
print("---------------------------------------------------------------------------------------------------------------")

# Check the Dates.
data['Date'] = pd.to_datetime(data['Date']).dt.date
print('Start Date:')
print(data['Date'].min())
print('\nEnd Date:')
print(data['Date'].max())
print('\nDays Count:')
print((data['Date'].max()-data['Date'].min()).days)
date_check = data.Date.diff()
print("---------------------------------------------------------------------------------------------------------------")
print('Row by Row Difference:')
print(date_check)
print('\nDifference Count:')
print(date_check.value_counts())

# Check Missing Values.
data['VIXClose'] = pd.to_numeric(data['VIXClose'], errors='coerce')  # Convert 'space' Character to 'NaN'.
print("---------------------------------------------------------------------------------------------------------------")
print('NaN Values:')
print('VIXClose:', data['VIXClose'].isnull().sum())

# Export Preprocessed Time Series.
print("---------------------------------------------------------------------------------------------------------------")
filename = 'data/VIXCloseData.csv'
print('\nData Exported to', filename, '\n')
data.to_csv(filename)
