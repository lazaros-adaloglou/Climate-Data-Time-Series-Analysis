# Imports.
import LinearAnalysisFunctions as lf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Get Path.
print("---------------------------------------------------------------------------------------------------------------")
print('Current Path', os.getcwd())

# Path to Time Series.
filename = 'data/Glasnevin_Data.csv'

# Preprocess Average Temperature Time Series.
try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
except FileNotFoundError:
    os.system('python GlasnevinDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
print("---------------------------------------------------------------------------------------------------------------")
print("Average Temperature Time Series of Glasnevin, Dublin:")
data = data.drop("Unnamed: 0", axis=1)
print(data)

# Define Average Temperature Time Series.
x = data.AvgTemp
savepath = 'data/'
value = 'AvgTemp'

# Plot Average Temperature.
lf.plot_timeseries(x, value, 'Average Temperature', savepath)
plt.show()

# Average Temperature Histogram.
lf.plot_histogram(x, value, 'Average Temperature', savepath)
plt.show()

plot_acf(x, zero=False, lags=10)
plt.show()

# Generate White Noise Data.
n = 1000
sd_noise = 1
mux = 0
xV = np.random.normal(0, sd_noise, n) + mux
