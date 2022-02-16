# Imports.
import LinearAnalysisFunctions as lf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima as pm
import datetime as dt
import pandas as pd
import numpy as np
import warnings
import os

# Suppress Warnings.
warnings.filterwarnings("ignore")

# Load Data.
filename = 'data/DublinAirport_Data.csv'
try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
except FileNotFoundError:
    os.system('python DublinAirportDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
print("---------------------------------------------------------------------------------------------------------------")
print("Average Temperature Time Series:\n")
savepath = 'data/'
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

# Average Temperature Histogram.
lf.plot_histogram(x, 'Frequency', 'Average Temperature Histogram', 'data/')
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
dates_year = range(1942, 2022)
plt.plot(dates_year, x_year, color='red', marker='x', linestyle='--', linewidth=1)
plt.xlabel('Time (Years)')
plt.ylabel('AvgTemp (°C)')
title = 'Yearly Mean Average Temperature'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Average Temperature Autocorrelation.
plot_acf(x, zero=False)
title = 'Autocorrelation'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Remove Trend.
# Polynomial Fit.
p = 40
pol = lf.polynomial_fit(x, p=p)

# Linear Breakpoint Fit.
p1 = 1
pol1 = []
breakpoints = 160
for i in range(0, 80 * 12 * 30, 180):
    pol1[i:i + 180] = lf.polynomial_fit(x[i:i + 180], p=p1)

# Plot Polynomial and Breakpoint Fit.
plt.plot(pol)
plt.plot(pol1)
plt.plot(x, alpha=0.5)
plt.xlim(15000, 18000)
plt.legend([f'Polynomial ({p})', f'Breakpoint ({breakpoints})', 'Original'])
title = f'Polynomial ({p}) and Breakpoint Fit ({breakpoints})'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Plot Polynomial and Linear Breakpoint Detrends.
plt.plot(x-pol, alpha=0.5)
plt.plot(x[0:28800]-pol1, alpha=0.5)
plt.legend([f'Polynomial ({p}) Detrended', f'Breakpoint ({breakpoints}) Detrended'])
title = f'Polynomial ({p}) vs Breakpoint ({breakpoints}) Detrended'
plt.title(title, x=0.5, y=1.0)
plt.xlim(15000, 18000)
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Moving Average Filter.
window = 105
ma = lf.rolling_window(x=x, window=window)
plt.plot(ma, linestyle='--')
plt.plot(x, alpha=0.5)
plt.legend([f'MA ({window})', 'Original'])
title = f'Moving Average ({window})'
plt.title(title, x=0.5, y=1.0)
plt.xlim(15000, 18000)
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# First Differences.
fd = np.diff(x)

# MA vs First Differences.
plt.plot(x-ma, alpha=0.5)
plt.plot(fd, alpha=0.5)
plt.xlim(15000, 16000)
title = f'First Differences vs MA ({window})'
plt.title(title, x=0.5, y=1.0)
plt.legend([f'MA ({window}) Detrended', 'First Differences Detrended'])
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# MA vs First Differences.
plt.plot(x-ma, alpha=0.3)
plt.plot(fd, alpha=0.3)
plt.xlim(15000, 16000)
title = f'First Differences vs MA ({window})'
plt.title(title, x=0.5, y=1.0)
plt.legend([f'MA ({window}) Detrended', 'First Differences Detrended'])
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Autocorrelation after Detrending with First Differences.
plot_acf(fd, zero=False)
title = 'Autocorrelation after Detrending with First Differences'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show()
