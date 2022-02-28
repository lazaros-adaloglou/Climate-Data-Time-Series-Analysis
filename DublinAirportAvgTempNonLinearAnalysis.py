# Imports.
from statsmodels.tsa.arima_process import arma_generate_sample, arma_acf, arma_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf
from nolitsa import delay, dimension, d2, utils
from sklearn.metrics import mean_squared_error
import NonLinearAnalysisFunctions as nlf
import LinearAnalysisFunctions as lf
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from statsmodels.api import OLS
from scipy.special import psi
from nolitsa import dimension
import statsmodels.api as sm
from scipy.stats import norm
import pmdarima as pm
import pandas as pd
import numpy as np
import statistics
import warnings
import nolds
import os

# Suppress Warnings.
warnings.filterwarnings("ignore")

# Load Data.
filename = 'Data/DublinAirport_Data.csv'
try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
except FileNotFoundError:
    os.system('python DublinAirportDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
print("===============================================================================================================")
print("Average Temperature Time Series:\n")
savepath = 'Figures/AirportLin'
data = data.drop("Unnamed: 0", axis=1)
print(data)
dates = data.date
date_axis = [d.to_pydatetime() for d in dates]

# Define Average Temperature Time Series.
x = data.AvgTemp.values
x_df = pd.DataFrame({data.AvgTemp.name: x})
m = len(x_df)

# Average Temperature Autocorrelation of Original Time Series.
plot_acf(x, zero=False)
title = 'Autocorrelation of Original Time Series'
plt.title(title, x=0.5, y=1.0)
plt.xlabel('lag(tau)')
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Stationary Time Series.
window = 92
ma = lf.rolling_window(x=x, window=window)
plt.figure()
plt.plot(ma, linestyle='--')
plt.plot(x, alpha=0.5)
plt.legend([f'MA ({window})', 'Original'])
title = f'Moving Average ({window})'
plt.xlabel('Time (Days)')
plt.ylabel('AvgTemp (°C)')
plt.title(title, x=0.5, y=1.0)
plt.xlim(15000, 18000)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)
fd = np.log(mas + abs(min(mas)) + 1)
fd = fd - fd.mean()
plt.figure()
plt.plot(fd, linestyle='--')
plt.legend(['Log(X_Detrended+abs(min(X_Detrended)) + 1)-mean'])
title = 'Logarithms of Detrended Average Temperature Time Series'
plt.xlabel('Time (Days)')
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)
plt.figure()
plt.plot(fd)
plt.legend(['Log(X_Detrended+abs(min(X_Detrended)) + 1)-mean'])
title = 'Logarithms of Detrended Time Series (Increased Resolution)'
plt.xlabel('Time (Days)')
plt.title(title, x=0.5, y=1.0)
plt.xlim(15000, 22000)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Hypothesis test for white noise after Detrending with MA (92) and taking the logs.
# Autocorrelation.
fd = fd[15000:16000]
maxtau = 31
acvf = lf.get_acf(fd, lags=maxtau)
title = 'Autocorrelation of log(X_detrended)'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Model Adaption and Forecasting Single Step and Multistep.
# AR Model.
# Partial Autocorrelation Criterion for choosing model order.
pacvf = lf.get_pacf(fd, lags=maxtau)
title = 'Partial Autocorrelation of log(X_detrended)'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)
plt.pause(0.001)

# Model AR(6).
p_ar = 6
summary_ar, fittedvalues_ar, resid_ar, model_ar, aic_ar = lf.fit_arima_model(x=fd, p=p_ar, q=0, savepath=savepath,
                                                                             d=0, show=True)
plt.pause(0.001)
print("===============================================================================================================")
print("Summary of chosen AR Model:\n")
print(summary_ar)
plt.pause(0.001)

# Portmanteau Test to see if the residuals are white noise.
lf.portmanteau_test(resid_ar, maxtau, best_p_ar, 0, 0, savepath, show=True)

n = 100
x0 = 0.51
r = 3.9
xV = nlf.logisticmap(n=n, r=r, x0=x0)
plt.figure()
plt.plot(xV)
# embedded = nlf.embed_data(xV, order=2, delay=1)

plt.scatter(xV[:-1], xV[1:])
xV = nlf.generate_arma_ts([0.8], [0], n)
plt.plot(xV)
plt.show()
embedded = nlf.embed_data(xV, order=2, delay=1)
print(embedded)

plt.scatter(xV[:-1], xV[1:])
xM = data.henon()
plt.plot(xM)
plt.figure(figsize=(14, 8))
xM = xM[:, 0]
plt.plot(xM)
embedded = nlf.embed_data(xM, order=3, delay=1)
nlf.plot_3d_attractor(embedded)

# REAL DATA ATTRACTORS
os.chdir('d:/timeserieslab/')
df = pd.read_csv('./data/BTCUSDT.csv')
df.set_index(pd.to_datetime(df['time']), inplace=True)
df.drop('time', axis=1, inplace=True)
df = df['01-01-2020':'12-31-2021']
df.plot()
logreturns = df.apply(lambda x: np.log(x)).diff().bfill()
logreturns.plot()
embedded = nlf.embed_data(logreturns.values.reshape(-1, ), order=2, delay=1)
plt.scatter(embedded[:-1], embedded[1:])
eeg = pd.read_csv('./data/epileeg.dat')
# eeg.plot();
embedded = nlf.embed_data(eeg.values.reshape(-1, ), order=3, delay=1)
nlf.plot_3d_attractor(embedded)
# plt.scatter(embedded[:-1], embedded[1:], linestyle='--', marker='x');

# Check for stationarity (Visual Inspection + statistical test)
xM = xM[2900:3000]
plt.figure(figsize=(14, 8))
plt.plot(xM)

adf = adfuller(xM, maxlag=None)
print(adf)

# Compute autocorrelation and delayed mutual information.
print(dir(delay))
help(delay.acorr)
# xM.shape

maxtau = 10
lag = np.arange(maxtau)
r = delay.acorr(xM, maxtau=maxtau)
i = delay.dmi(xM, maxtau=maxtau)
r_delay = np.argmax(r < 1.0 / np.e)

plt.figure(1, figsize=(14, 8))

plt.subplot(211)
plt.title(r'Delay estimation for Henon map')
plt.ylabel(r'Delayed mutual information')
plt.plot(lag, i, marker='o')

plt.subplot(212)
plt.xlabel(r'Time delay $\tau$')
plt.ylabel(r'Autocorrelation')
# plt.plot(lag, r, r_delay, r[r_delay], 'o')
plt.axhline(1.0 / np.e, linestyle='--', alpha=0.7, color='red')

plt.figure(2, figsize=(14, 8))
plt.subplot(111)
plt.title(r'Time delay = %d' % r_delay)
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(xM[:-r_delay], xM[r_delay:], '.')

plt.show()
print(r)
print(r'Autocorrelation time = %d' % r_delay)

dim = np.arange(1, 10 + 1)
f1, f2, f3 = dimension.fnn(xM, tau=1, dim=dim, window=10, metric='cityblock')

plt.title(r'FNN for Henon map')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()
plt.show()

# Correlation Sum
plt.figure(figsize=(14, 8))
plt.title('Local $D_2$ vs $r$ for Henon map')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')
theiler_window = 10
tau = 1
dim = np.arange(1, 10 + 1)

for r, c in d2.c2_embed(xM, tau=tau, dim=dim, window=theiler_window,
                        r=utils.gprange(0.001, 1.0, 100)):
    plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')

plt.semilogx(utils.gprange(0.001, 1.0, 100), 1.220 * np.ones(100),
             color='#000000')
plt.show()

r = utils.gprange(0.001, 1.0, 100)
corr_dim, debug_data = nolds.corr_dim(xM, emb_dim=2, rvals=r, debug_data=True)

# Values used for log(r).
rvals = debug_data[0]

# The corresponding log(C(r)).
csums = debug_data[1]

# Line coefficients ([slope, intercept]).
poly = debug_data[2]

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111)
ax.plot(rvals, csums)
ax.set_xlabel('log(r)')
ax.set_ylabel('log(C(r))')

# Predictions
# - LAP
# - LLP
n = xM.shape[0]
test_prop = 0.3
split_point = int(n * (1 - test_prop))

train_xM = xM[:split_point]
test_xM = xM[split_point:]

plt.figure(figsize=(16, 8))
plt.plot(np.arange(split_point), train_xM, color='blue', alpha=0.7, label='train set')
plt.plot(np.arange(split_point, xM.shape[0]), test_xM, color='red', linestyle='--', alpha=0.7, label='test set')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')

embed_train_data = nlf.embed_data(train_xM, 2, 1)
embed_test_data = nlf.embed_data(test_xM, 2, 1)

knn = 5
tree = KDTree(embed_train_data, leaf_size=1, metric='chebyshev')
neighbors_idx = []
for i, state in enumerate(embed_test_data):
    dist, neigh_idx = tree.query(state.reshape(1, -1), k=knn)
    neighbors_idx.append(tuple([i, neigh_idx[0]]))

plt.figure(figsize=(16, 8))
plt.plot(xM)
plt.xlabel('time')
plt.ylabel('xM')
plt.axvline(split_point, linestyle='--', color='black', alpha=0.5)
# Get neighbors of first test datapoint.
check_test_point = 1
neigh = neighbors_idx[check_test_point]
print(neigh)
test_state_idx = neigh[0]
neighs_idx = neigh[1]
plt.plot([neighs_idx, neighs_idx + 1], [xM[neighs_idx], xM[neighs_idx + 1]], linestyle='--', color='orange')
plt.scatter([neighs_idx, neighs_idx + 1], [xM[neighs_idx], xM[neighs_idx + 1]], linestyle='--', color='orange')
# Test Set State.
plt.plot([test_state_idx + split_point, test_state_idx + split_point + 1],
         [xM[test_state_idx + split_point], xM[test_state_idx + split_point + 1]], linestyle='--', color='red')
plt.scatter([test_state_idx + split_point, test_state_idx + split_point + 1],
            [xM[test_state_idx + split_point], xM[test_state_idx + split_point + 1]], linestyle='--', color='red')

plt.figure(figsize=(16, 8))
plt.scatter(embed_train_data[:, 0], embed_train_data[:, 1], alpha=0.3)
plt.scatter(embed_train_data[neighs_idx][:, 0], embed_train_data[neighs_idx][:, 1], label=f'neighbors in train data')
plt.scatter(embed_test_data[test_state_idx, 0], embed_test_data[test_state_idx, 1],
            label=f'test point:{test_state_idx}')
plt.xlabel('x(t)')
plt.ylabel('x(t+1)')
plt.legend()

T = 1
lap_predictions = []
for neigh in neighbors_idx:
    test_state_idx = neigh[0]
    neighs_idx = neigh[1]
    images_idx = neighs_idx + T
    images = xM[images_idx]
    lap = np.sum(images) / len(images)
    lap_predictions.append(lap)
plt.figure(figsize=(14, 8))
plt.plot(np.arange(split_point + 1, xM.shape[0]), lap_predictions, label='LAP prediction', alpha=0.9, linestyle='-.')
plt.plot(np.arange(split_point + 1, xM.shape[0]), test_xM[1:], label='True values', alpha=0.9, linestyle='--')

np.sqrt(np.mean((np.array(lap_predictions) - test_xM[1:]) ** 2)) / np.std(test_xM[1:])

X, y = [], []
for neigh in neighs_idx:
    x_ = [xM[neigh], xM[neigh + 1]]
    y_ = [xM[neigh + 1 + T]]
    X.append(x_)
    y.append(y_)

X = sm.add_constant(X)
X = np.asarray(X)
y = np.asarray(y)
ols = sm.OLS(endog=y, exog=X).fit()
llp = np.dot(ols.params, [1, xM[i + split_point], xM[i + split_point + 1]])
ols.summary()

# Real Data
with open('epileeg.dat', 'r') as file:
    lines = file.readlines()
xM = np.full(shape=len(lines), fill_value=np.nan)
for i, line in enumerate(lines):
    point = line.rstrip().lstrip()
    xM[i] = point
xM = np.array(xM)
plt.plot(xM)
plt.show()

