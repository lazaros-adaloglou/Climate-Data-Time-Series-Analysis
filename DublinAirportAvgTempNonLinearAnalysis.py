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
savepath = 'Figures/AirportNonLin'
data = data.drop("Unnamed: 0", axis=1)
print(data)
dates = data.date
date_axis = [d.to_pydatetime() for d in dates]

# Define Average Temperature Time Series.
x = data.AvgTemp.values
x_df = pd.DataFrame({data.AvgTemp.name: x})
m = len(x_df)

# Stationary Time Series.
window = 92
ma = lf.rolling_window(x=x, window=window)
mas = x - ma
fd = np.log(mas + abs(min(mas)) + 1)
fd = fd - fd.mean()
fd = fd[15000:18000]

# Model AR(6).
p_ar = 6
summary_ar, fittedvalues_ar, resid_ar, model_ar, aic_ar = lf.fit_arima_model(x=fd, p=p_ar, q=0, savepath=savepath,
                                                                             d=0, show=False)

# Residuals.
xV = resid_ar
plt.figure()
plt.plot(xV)
plt.legend(['Residuals'])
title = f'Residuals of AR({p_ar})'
plt.xlabel('Time (Days)')
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)
plt.pause(0.001)

# Portmanteau Test to see if the residuals are white noise.
maxtau = 31
lf.portmanteau_test(resid_ar, maxtau, p_ar, 0, 0, savepath, show=True)

# Create 40 new TS of same length without correlations with random permutations of the n-length residuals.
# This is done to explore if there are any non-linear correlations. (residuals=white noise,
# but are they iid, or have they non-linear correlations?) To have non-linear correlations,
# the statistics we choose must have a little differences in the 40 tseries. If q=linear, then q0 belongs
# in the distribution of q1...140. For statistics that accept different input parameters, like correlation,
# AR order, embedding dimension and the number of neighbors for the local model, we need to try different values.
tot = range(0, 40)
ts = []
for i in tot:
    ts.append(np.random.permutation(fd))

# Check for stationarity Dickey-Fuller Statistical Test (if p-value<0.05 -> H0 rejected -> stationarity).
adf = [adfuller(xV, maxlag=None)[1]]
for i in tot:
    adf.append(adfuller(ts[i], maxlag=None)[1])
plt.figure()
plt.plot(adf)
plt.legend(['Dickey-Fuller Test p-values'])
title = 'Dickey-Fuller Test for Stationarity (if H0 rejected) on Original + 40 generated time series'
plt.xlabel('Time Series')
plt.ylabel('p-values')
plt.title(title)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=True)
plt.pause(0.001)

# Compute autocorrelation and delayed mutual information.
# for initial time series
maxtau = 10
lag = np.arange(maxtau)
r = delay.acorr(xV, maxtau=maxtau)
i = delay.dmi(xV, maxtau=maxtau)
r_delay = np.argmax(r < 1.0 / np.e)
plt.figure(1, figsize=(14, 8))
plt.subplot(211)
title = r'Delay estimation of original time series'
plt.title(title)
plt.ylabel(r'Delayed mutual information of original time series')
plt.plot(lag, i, marker='o')
plt.subplot(212)
plt.xlabel(r'Time delay $\tau$ of original time series')
plt.ylabel(r'Autocorrelation of original time series')
plt.plot(lag, r, r_delay, r[r_delay], 'o')
plt.axhline(1.0 / np.e, linestyle='--', alpha=0.7, color='red')
plt.savefig(f'{savepath}/{title}.png')
plt.show()
plt.pause(0.001)

# for 40 time series.
plt.figure(1, figsize=(14, 8))
title = r'Delay estimation of original + 40 time series'
plt.title(title)
plt.ylabel(r'Delayed mutual information of original + 40 time series')
r = [r]
i = [i]
r_delay = [r_delay]
for s in tot:
    i.append(delay.dmi(ts[s], maxtau=maxtau))
    plt.plot(lag, i[s])
plt.savefig(f'{savepath}/{title}.png')
plt.show()

# Compute Dimension.
# for initial time series.
# plt.figure(1, figsize=(14, 8))
# dim = np.arange(1, 10 + 1)
# f1 = nlf.falsenearestneighbors(xV, m_max=10, tau=1, show=False)
# f1, _, _ = dimension.fnn(xV, tau=1, dim=dim, window=10, metric='cityblock')
# title = r'FNN for original time series'
# plt.title(title)
# plt.xlabel(r'Embedding dimension $d$')
# plt.ylabel(r'FNN (%)')
# plt.plot(dim, 100 * f1, 'bo--')
# plt.legend(r'Test I')
# plt.savefig(f'{savepath}/{title}.png')
# plt.show(block=True)

# # Correlation Sum
# plt.figure(figsize=(14, 8))
# plt.title('Local $D_2$ vs $r$ for original time series')
# plt.xlabel(r'Distance $r$')
# plt.ylabel(r'Local $D_2$')
# theiler_window = 10
# tau = 1
# dim = np.arange(1, 10 + 1)
# for r, c in d2.c2_embed(xM, tau=tau, dim=dim, window=theiler_window, r=utils.gprange(0.001, 1.0, 100)):
#     plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')
# plt.semilogx(utils.gprange(0.001, 1.0, 100), 1.220 * np.ones(100), color='#000000')
# plt.show()

# r = utils.gprange(0.001, 1.0, 100)
# corr_dim, debug_data = nolds.corr_dim(xM, emb_dim=2, rvals=r, debug_data=True)
#
# # Values used for log(r).
# rvals = debug_data[0]
#
# # The corresponding log(C(r)).
# csums = debug_data[1]
#
# # Line coefficients ([slope, intercept]).
# poly = debug_data[2]
#
# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111)
# ax.plot(rvals, csums)
# ax.set_xlabel('log(r)')
# ax.set_ylabel('log(C(r))')

# # Embed and Plot 3D Attractor.
# embedded = nlf.embed_data(xV, m=3, tau=1)
# print(embedded)
# nlf.plot_3d_attractor(embedded)
# plt.scatter(embedded[:-1], embedded[1:], linestyle='--', marker='x')
# plt.show()

# # Predictions
# # - LAP
# # - LLP
# n = xM.shape[0]
# test_prop = 0.3
# split_point = int(n * (1 - test_prop))
#
# train_xM = xM[:split_point]
# test_xM = xM[split_point:]
#
# plt.figure(figsize=(16, 8))
# plt.plot(np.arange(split_point), train_xM, color='blue', alpha=0.7, label='train set')
# plt.plot(np.arange(split_point, xM.shape[0]), test_xM, color='red', linestyle='--', alpha=0.7, label='test set')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Value')
#
# embed_train_data = nlf.embed_data(train_xM, 2, 1)
# embed_test_data = nlf.embed_data(test_xM, 2, 1)
#
# knn = 5
# tree = KDTree(embed_train_data, leaf_size=1, metric='chebyshev')
# neighbors_idx = []
# for i, state in enumerate(embed_test_data):
#     dist, neigh_idx = tree.query(state.reshape(1, -1), k=knn)
#     neighbors_idx.append(tuple([i, neigh_idx[0]]))
#
# plt.figure(figsize=(16, 8))
# plt.plot(xM)
# plt.xlabel('time')
# plt.ylabel('xM')
# plt.axvline(split_point, linestyle='--', color='black', alpha=0.5)
# # Get neighbors of first test datapoint.
# check_test_point = 1
# neigh = neighbors_idx[check_test_point]
# print(neigh)
# test_state_idx = neigh[0]
# neighs_idx = neigh[1]
# plt.plot([neighs_idx, neighs_idx + 1], [xM[neighs_idx], xM[neighs_idx + 1]], linestyle='--', color='orange')
# plt.scatter([neighs_idx, neighs_idx + 1], [xM[neighs_idx], xM[neighs_idx + 1]], linestyle='--', color='orange')
# # Test Set State.
# plt.plot([test_state_idx + split_point, test_state_idx + split_point + 1],
#          [xM[test_state_idx + split_point], xM[test_state_idx + split_point + 1]], linestyle='--', color='red')
# plt.scatter([test_state_idx + split_point, test_state_idx + split_point + 1],
#             [xM[test_state_idx + split_point], xM[test_state_idx + split_point + 1]], linestyle='--', color='red')
#
# plt.figure(figsize=(16, 8))
# plt.scatter(embed_train_data[:, 0], embed_train_data[:, 1], alpha=0.3)
# plt.scatter(embed_train_data[neighs_idx][:, 0], embed_train_data[neighs_idx][:, 1], label=f'neighbors in train data')
# plt.scatter(embed_test_data[test_state_idx, 0], embed_test_data[test_state_idx, 1],
#             label=f'test point:{test_state_idx}')
# plt.xlabel('x(t)')
# plt.ylabel('x(t+1)')
# plt.legend()
#
# T = 1
# lap_predictions = []
# for neigh in neighbors_idx:
#     test_state_idx = neigh[0]
#     neighs_idx = neigh[1]
#     images_idx = neighs_idx + T
#     images = xM[images_idx]
#     lap = np.sum(images) / len(images)
#     lap_predictions.append(lap)
# plt.figure(figsize=(14, 8))
# plt.plot(np.arange(split_point + 1, xM.shape[0]), lap_predictions, label='LAP prediction', alpha=0.9, linestyle='-.')
# plt.plot(np.arange(split_point + 1, xM.shape[0]), test_xM[1:], label='True values', alpha=0.9, linestyle='--')
#
# np.sqrt(np.mean((np.array(lap_predictions) - test_xM[1:]) ** 2)) / np.std(test_xM[1:])
#
# X, y = [], []
# for neigh in neighs_idx:
#     x_ = [xM[neigh], xM[neigh + 1]]
#     y_ = [xM[neigh + 1 + T]]
#     X.append(x_)
#     y.append(y_)
#
# X = sm.add_constant(X)
# X = np.asarray(X)
# y = np.asarray(y)
# ols = sm.OLS(endog=y, exog=X).fit()
# llp = np.dot(ols.params, [1, xM[i + split_point], xM[i + split_point + 1]])
# ols.summary()

