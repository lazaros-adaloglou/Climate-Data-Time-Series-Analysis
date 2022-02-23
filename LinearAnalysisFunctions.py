# Imports.
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


# Plot Time Series.
def plot_timeseries(x, value='', title='', savepath='', dates=None, zoomx=False, color='C0'):
    plt.figure()
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
    plt.figure()
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


# Calculate pacf of timeseries xV to lag (lags) and show figure with confidence interval with (alpha).
def get_pacf(xv, lags=10, alpha=0.05, show=True):
    pacfv = pacf(xv, nlags=lags)[1:]
    z_inv = norm.ppf(1 - alpha / 2)
    upperbound95 = z_inv / np.sqrt(xv.shape[0])
    lowerbound95 = -upperbound95
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, lags + 1), pacfv, marker='o')
        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1 - alpha) * 100}%')
        ax.axhline(lowerbound95, linestyle='--', color='red')
        ax.set_title('PACF')
        ax.set_xlabel('Lag')
        ax.set_xticks(np.arange(1, lags + 1))
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        ax.legend()
    return pacfv


# PORTMANTEAULB hypothesis test (H0) for independence of time series: tests jointly that several autocorrelations
# are zero. It computes the Ljung-Box statistic of the modified sum of autocorrelations up to a maximum lag, for
# maximum lags 1,2,...,maxtau.
def portmanteau_test(xv, maxtau, p, d, q, show=False):
    df = acorr_ljungbox(xv, lags=maxtau)
    lbpv = df.lb_pvalue.values
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(np.arange(len(lbpv)), lbpv)
        ax.axhline(0.05, linestyle='--', color='r')
        title = f'Ljung-Box Portmanteau Test for Residuals of ARIMA({p},{d},{q})'
        ax.set_title(title)
        plt.savefig(f'{savepath}/{title}.png')
        ax.set_yticks(np.arange(0, 1.1))
        plt.show(block=False)
        plt.pause(0.001)
    return lbpv


# Fit ARIMA(p, d, q) in x returns: summary (table), fittedvalues, residuals, model, AIC.
def fit_arima_model(x, p, q, d=0, show=False):
    model = ARIMA(x, order=(p, d, q)).fit()
    summary = model.summary()
    fittedvalues = model.fittedvalues
    fittedvalues = np.array(fittedvalues).reshape(-1, 1)
    resid = model.resid
    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8))
        ax1.plot(x, label='Original', color='blue')
        ax1.plot(fittedvalues, label='Fitted Values', color='red', linestyle='--', alpha=0.9)
        ax1.legend()
        ax1.set_title(f'ARIMA({p}, {d}, {q})')
        ax2.scatter(np.arange(len(resid)), resid, label='Residuals')
        ax2.legend()
        ax3.hist(resid)
        ax3.set_ylabel('Frequency')
        ax3.set_xlabel('Residuals Histogram')
        title = f'ARIMA({p},{d},{q})'
        plt.savefig(f'Figures/{title}.png')
        plt.show(block=False)
        plt.pause(0.001)
    return summary, fittedvalues, resid, model, model.aic


# Calculate fitting error with NRMSE for given model in timeseries x till prediction horizon Tmax. Returns: nrmsev
# preds: for timesteps T=1, 2, 3.
def calculate_fitting_error(x, model, tmax=20, show=False):
    nrmsev = np.full(shape=tmax, fill_value=np.nan)
    nobs = len(x)
    xvstd = np.std(x)
    # vartar = np.sum((x - np.mean(x)) ** 2)
    predm = []
    tmin = np.max([len(model.arparams), len(model.maparams), 1])
    # Start prediction after getting all lags needed from model.
    for T in np.arange(1, tmax):
        errors = []
        predv = np.full(shape=nobs, fill_value=np.nan)
        for t in np.arange(tmin, nobs - T):
            pred_ = model.predict(start=t, end=t + T - 1, dynamic=True)
            # predv.append(pred_[-1])
            ytrue = x[t + T - 1]
            predv[t + T - 1] = pred_[-1]
            error = pred_[-1] - ytrue
            errors.append(error)
        predm.append(predv)
        errors = np.array(errors)
        mse = np.mean(np.power(errors, 2))
        rmse = np.sqrt(mse)
        nrmsev[T] = (rmse / xvstd)
        # nrmsev[T] = (np.sum(errors**2) / vartar)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, tmax), nrmsev[1:], marker='x', label='NRMSE')
        ax.axhline(1, color='red', linestyle='--')
        p = len(model.arparams) - 2
        q = len(model.maparams) - 2
        d = 0
        title = f'Fitting Error of ARIMA({p},{d},{q}) for T = {tmax}'
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel('T')
        ax.set_xticks(np.arange(1, tmax))
        plt.show()
        # Plot multistep prediction for T=1, 2, 3
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(x, label='original')
        colors = ['red', 'green', 'black']
        for i, preds in enumerate(predm[:3]):
            ax.plot(preds, color=colors[i], linestyle='--', label=f'T={i + 1}', alpha=0.7)
        ax.legend(loc='best')
        plt.savefig(f'Figures/{title}.png')
        plt.show(block=False)
        plt.pause(0.001)
    return nrmsev, predm


# Multistep oos prediction (out of sample predictions starting from last train values).
def predict_oos_multistep(model, tmax=10, return_conf_int=True, alpha=0.05, show=True):
    if return_conf_int:
        preds, conf_bounds = model.predict(n_periods=tmax, return_conf_int=return_conf_int, alpha=alpha)
    else:
        preds = model.predict(n_periods=tmax, return_conf_int=return_conf_int, alpha=alpha)
        conf_bounds = []
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, tmax+1), preds)
    return preds, conf_bounds
