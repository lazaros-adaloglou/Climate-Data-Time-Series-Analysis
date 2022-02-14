# Imports.
import LinearAnalysisFunctions as lf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima as pm
import pandas as pd
import numpy as np
import os

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

# Define Average Temperature Time Series.
x = data.AvgTemp.values
x_df = pd.DataFrame({data.AvgTemp.name: x})
strings = ['AvgTemp (°C)', 'Average Temperature', 'data/']

# Plot Average Temperature.
lf.plot_timeseries(x, strings[0], strings[1], strings[2])
plt.show()

# Average Temperature Histogram.
lf.plot_histogram(x, strings[0], strings[1], strings[2])
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



