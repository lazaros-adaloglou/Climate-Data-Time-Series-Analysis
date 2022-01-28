# Imports.
import LinearAnalysisFunctions as lf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
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
print("Time Series:")
data = data.drop("Unnamed: 0", axis=1)
print(data)

# Define Average Temperature Time Series.
x = data.AvgTemp
savepath = 'data/'
value = 'AvgTemp (°C)'

# Plot Average Temperature.
lf.plot_timeseries(x, value, 'Average Temperature', savepath)
plt.show()

# Average Temperature Histogram.
lf.plot_histogram(x, value, 'Average Temperature', savepath)
plt.show()

# Remove Trend
