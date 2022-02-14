# Imports.
import LinearAnalysisFunctions as lf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima as pm
import datetime as dt
import pandas as pd
import numpy as np
import os

# Load Data.
filename = 'data/DublinAirport_Data.csv'
try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
except FileNotFoundError:
    os.system('python DublinAirportDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
print("---------------------------------------------------------------------------------------------------------------")
print("Average Temperature Time Series:\n")
data = data.drop("Unnamed: 0", axis=1)
print(data)
dates = data.date
print(dates)
date_axis = [d.to_pydatetime() for d in dates]

# Define Average Temperature Time Series.
x = data.AvgTemp.values
x_df = pd.DataFrame({data.AvgTemp.name: x})

# Plot Average Temperature.
lf.plot_timeseries(x, 'AvgTemp (°C)', 'Average Temperature', 'data/', date_axis)
plt.show()

# Plot Zoomed Average Temperature.
lf.plot_timeseries(x, 'AvgTemp (°C)', 'Average Temperature (1998-2014)', 'data/', date_axis, zoomx=True)
plt.show()

# Plot Yearly Mean Average Temperature.
x_year = []
years = 80
days = 365
for year in range(0, years):
    k = 0
    for day in range(1, days + 1):
        k = k + x[year * days + day]
    x_year.append(k/days)
lf.plot_timeseries(x_year, 'AvgTemp (°C)', 'Yearly Mean Average Temperature', 'data/', color='red')
plt.xlabel('Time (Years)')
plt.show()

# Average Temperature Histogram.
lf.plot_histogram(x, 'Frequency', 'Average Temperature Histogram', 'data/')
plt.show()

# Average Temperature Autocorrelation.
plt.show()

# Remove Trend.
window = 15
ma = lf.rolling_window(x=x, window=window)
plt.plot(ma, linestyle='--')
plt.plot(x, alpha=0.5)
plt.plot(x-ma, alpha=0.5)
plt.legend([f'MA', 'Original', 'Detrended'])
plt.show()

# First Differences.
fd = np.diff(x)

# Comparison.
plt.plot(x-ma, alpha=0.5)
plt.plot(fd, alpha=0.5)
plt.xlim(1000, 2000)
plt.legend(['ma', 'diffs'])
plt.show()

# Correlation.
plot_acf(x-ma, zero=False)
plt.show()



