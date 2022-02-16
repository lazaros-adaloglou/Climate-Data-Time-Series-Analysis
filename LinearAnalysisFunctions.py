from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


# Plot Time Series.
def plot_timeseries(x, value='', title='', savepath='', dates=None, zoomx=False, color='C0'):

    if dates is not None:
        plt.plot(dates, x, color=color, marker='x', linestyle='--', linewidth=1)
        plt.gcf().autofmt_xdate()
    else:
        plt.plot(x, color=color, marker='x', linestyle='--', linewidth=1)
        plt.xlabel('Time (Days)')
    plt.ylabel(value)
    if zoomx is True:
        left, right = plt.xlim()
        plt.xlim(right/2, right/2+right/3.5)
    if len(title) > 0:
        plt.title(title, x=0.5, y=1.0)
    if len(savepath) > 0:
        plt.savefig(f'{savepath}/{title} Time Series.png')


# Plot Histogram.
def plot_histogram(x, value, title='', savepath=''):

    plt.hist(x, alpha=0.8, rwidth=0.9)
    plt.xlabel(value)
    plt.title('Histogram')
    if len(title) > 0:
        plt.title(title, x=0.5, y=1.0)
    if len(savepath) > 0:
        plt.savefig(f'{savepath}/{title}.png')


# Returns the Moving Average of a Time Series x with Length of Window.
def rolling_window(x, window):

    x = x.flatten()
    return np.convolve(x, np.ones(window) / window, mode='same')


# Fit to a given time series with a polynomial of a given order. x: vector of length 'n' of the time series
# p: the order of the polynomial to be fitted. return: vector of length 'n' of the fitted time series.
def polynomial_fit(x, p):

    n = x.shape[0]
    x = x[:]
    if p > 0:
        tv = np.arange(n)
        bv = np.polyfit(x=tv, y=x, deg=p)
        muv = np.polyval(p=bv, x=tv)
    else:
        muv = np.full(shape=n, fill_value=np.nan)
    return muv

