from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
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
    plt.ylabel('Frequency')
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


# Calculate acf of timeseries xV to lag (lags) and show figure with confidence interval with (alpha).
def get_acf(x, lags=10, alpha=0.05, show=True):

    acfv = acf(x, nlags=lags)[1:]
    z_inv = norm.ppf(1 - alpha / 2)
    upperbound95 = z_inv / np.sqrt(x.shape[0])
    lowerbound95 = -upperbound95
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, lags + 1), acfv, marker='o')
        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1 - alpha) * 100}%')
        ax.axhline(lowerbound95, linestyle='--', color='red')
        ax.set_title('ACF')
        ax.set_xlabel('Lag')
        ax.set_xticks(np.arange(1, lags + 1))
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        ax.legend()
    return acfv


# Computes the periodic time series comprised of repetetive patterns of seasonal components given a time series and
# the season (period).
def seasonal_components(x, period):

    n = x.shape[0]
    sv = np.full(shape=(n,), fill_value=np.nan)
    monv = np.full(shape=(period,), fill_value=np.nan)
    for i in np.arange(period):
        monv[i] = np.mean(x[i:n:period])
    monv = monv - np.mean(monv)
    for i in np.arange(period):
        sv[i:n:period] = monv[i] * np.ones(shape=len(np.arange(i, n, period)))
    return sv


# PORTMANTEAULB hypothesis test (H0) for independence of time series: tests jointly that several autocorrelations
# are zero. It computes the Ljung-Box statistic of the modified sum of autocorrelations up to a maximum lag, for
# maximum lags 1,2,...,maxtau.
def portmanteau_test(x, maxtau, show=False):

    ljung_val, ljung_pval = acorr_ljungbox(x, lags=maxtau)
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(np.arange(len(ljung_pval)), ljung_pval)
        ax.axhline(0.05, linestyle='--', color='r')
        ax.set_title('Ljung-Box Portmanteau test')
        ax.set_yticks(np.arange(0, 1.1))
        plt.show()
    return ljung_val, ljung_pval

