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
filename = 'data/VIXCloseData.csv'
try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['Date'])
except FileNotFoundError:
    os.system('python VIXDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['Date'])
print("---------------------------------------------------------------------------------------------------------------")
print("VIX Close Time Series:\n")
savepath = 'data/'
data = data.drop("Unnamed: 0", axis=1)
print(data)
dates = data.Date
print(dates)
date_axis = [d.to_pydatetime() for d in dates]

# Define VIX Close Time Series.
x = data.VIXClose.values
x_df = pd.DataFrame({data.VIXClose.name: x})

# Plot VIX.
lf.plot_timeseries(x, 'VIX Close', 'VIX Close', 'data/', date_axis)
plt.show()

# VIX Close Histogram.
lf.plot_histogram(x, 'Frequency', 'VIX Close Histogram', 'data/')
plt.show()

# Plot Zoomed VIX Close.
lf.plot_timeseries(x, 'VIX Close', 'VIX Close (-)', 'data/', date_axis, zoomx=True)
plt.show()

# # Plot Yearly Mean VIX Close.
# x_year = []
# years = 80
# days = 365
# for year in range(0, years):
#     k = 0
#     for day in range(1, days + 1):
#         k = k + x[year * days + day]
#     x_year.append(k/days)
# dates_year = range(1942, 2022)
# plt.plot(dates_year, x_year, color='red', marker='x', linestyle='--', linewidth=1)
# plt.xlabel('Time (Years)')
# plt.ylabel('AvgTemp (°C)')
# title = 'Yearly Mean Average Temperature'
# plt.title(title, x=0.5, y=1.0)
# plt.savefig(f'{savepath}/{title}.png')
# plt.show()

# VIX Close Autocorrelation.
plot_acf(x, zero=False)
title = 'Autocorrelation'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Remove Trend.
# Polynomial Fit.
p = 40
pol = lf.polynomial_fit(x, p=p)

# # Linear Breakpoint Fit.
# p1 = 1
# pol1 = []
# breakpoints = 160
# for i in range(0, 80 * 12 * 30, 180):
#     pol1[i:i + 180] = lf.polynomial_fit(x[i:i + 180], p=p1)
#
# # Plot Polynomial and Breakpoint Fit.
# plt.plot(pol)
# plt.plot(pol1)
# plt.plot(x, alpha=0.5)
# plt.xlim(15000, 18000)
# plt.legend([f'Polynomial ({p})', f'Breakpoint ({breakpoints})', 'Original'])
# title = f'Polynomial ({p}) and Breakpoint Fit ({breakpoints})'
# plt.title(title, x=0.5, y=1.0)
# plt.savefig(f'{savepath}/{title}.png')
# plt.show()

# Plot Polynomial and Linear Breakpoint Detrends.
# plt.plot(x-pol, alpha=0.5)
# plt.plot(x[0:28800]-pol1, alpha=0.5)
# plt.legend([f'Polynomial ({p}) Detrended', f'Breakpoint ({breakpoints}) Detrended'])
# title = f'Polynomial ({p}) vs Breakpoint ({breakpoints}) Detrended'
# plt.title(title, x=0.5, y=1.0)
# plt.xlim(15000, 18000)
# plt.savefig(f'{savepath}/{title}.png')
# plt.show()

# Moving Average Filter.
window = 105
ma = lf.rolling_window(x=x, window=window)
plt.plot(ma, linestyle='--')
plt.plot(x, alpha=0.5)
plt.legend([f'MA ({window})', 'Original'])
title = f'Moving Average ({window})'
plt.title(title, x=0.5, y=1.0)
plt.xlim(2000, 3000)
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Differences of Logarithms.
logx = np.log(x)
fd = np.diff(logx)

# MA vs Differences of Logarithms.
plt.plot(x-ma, alpha=0.5)
plt.plot(fd, alpha=0.5)
plt.xlim(2000, 3000)
title = f'Differences of Logarithms vs MA ({window})'
plt.title(title, x=0.5, y=1.0)
plt.legend([f'MA ({window}) Detrended', 'Differences of Logarithms Detrended'])
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Autocorrelation after Detrending with Differences of Logarithms.
plot_acf(fd, zero=False)
acvf = lf.get_acf(fd)
# title = 'Autocorrelation after Detrending with Differences of Logarithms'
# plt.title(title, x=0.5, y=1.0)
# plt.savefig(f'{savepath}/{title}.png')
plt.show()
