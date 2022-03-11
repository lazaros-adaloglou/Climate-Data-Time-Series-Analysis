# Imports.
from statsmodels.graphics.tsaplots import plot_acf
import LinearAnalysisFunctions as lf
from matplotlib import pyplot as plt
import pmdarima as pm
import pandas as pd
import numpy as np
import statistics
import warnings
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

# Plot Average Temperature.
lf.plot_timeseries(x, 'AvgTemp (°C)', 'Average Temperature', savepath, date_axis)
plt.show(block=False)

# Average Temperature Histogram.
lf.plot_histogram(x, 'AvgTemp (°C)', 'Average Temperature Histogram', savepath)
plt.show(block=False)

# Plot Zoomed Average Temperature.
lf.plot_timeseries(x, 'AvgTemp (°C)', 'Average Temperature (1998-2014)', savepath, date_axis, zoomx=True)
plt.show(block=False)

# Plot Yearly Mean of Original Time Series.
x_year = []
years = 80
days = 365
for year in range(0, years):
    k = 0
    for day in range(1, days + 1):
        k = k + x[year * days + day]
    x_year.append(k/days)
dates_year = range(1942, 2022)
plt.figure()
plt.plot(dates_year, x_year, color='red', marker='x', linestyle='--', linewidth=1)
plt.xlabel('Time (Years)')
plt.ylabel('AvgTemp (°C)')
title = 'Yearly Mean of Original Time Series'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Average Temperature Autocorrelation of Original Time Series.
plot_acf(x, zero=False)
title = 'Autocorrelation of Original Time Series'
plt.title(title, x=0.5, y=1.0)
plt.xlabel('lag(tau)')
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

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

# Plot Polynomial and Breakpoint Fit
plt.figure()
plt.plot(pol)
plt.plot(pol1)
plt.plot(x, alpha=0.5)
plt.xlim(15000, 18000)
plt.legend([f'Polynomial ({p})', f'Breakpoint ({breakpoints})', 'Original'])
title = f'Polynomial ({p}) and Breakpoint Fit ({breakpoints})'
plt.xlabel('Time (Days)')
plt.ylabel('AvgTemp (°C)')
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Plot Polynomial and Linear Breakpoint Detrends.
plt.figure()
plt.plot(x-pol, alpha=0.5)
plt.plot(x[0:28800]-pol1, alpha=0.5)
plt.legend([f'Polynomial ({p}) Detrended', f'Breakpoint ({breakpoints}) Detrended'])
title = f'Polynomial ({p}) vs Breakpoint ({breakpoints}) Detrended'
plt.xlabel('Time (Days)')
plt.ylabel('AvgTemp (°C)')
plt.title(title, x=0.5, y=1.0)
plt.xlim(15000, 18000)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Moving Average Filter.
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

# Differences of Logarithms (Increment each zero with 0.1).
logx = []
lx = x
for i in range(0, len(lx)):
    if lx[i] == 0:
        lx[i] = lx[i] + 0.1
    if lx[i] < 0:
        logx.append(-np.log(abs(lx[i])))
    else:
        logx.append(np.log(lx[i]))
fd = np.diff(logx)

# MA vs Differences of Logarithms.
ma = lf.rolling_window(x, window)
plt.figure()
plt.plot(x-ma, alpha=0.5)
plt.plot(fd, alpha=0.5)
plt.xlim(15000, 16000)
title = f'Differences of Logarithms vs MA ({window})'
plt.title(title, x=0.5, y=1.0)
plt.xlabel('Time (Days)')
plt.ylabel('AvgTemp (°C)')
plt.legend([f'MA ({window}) Detrended', 'Differences of Logarithms Detrended'])
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Print Yearly Mean Average Temperature of detrended and not time series.
mas = x - ma
mas_year = []
years = 80
days = 365
for year in range(0, years):
    k = 0
    for day in range(1, days + 1):
        k = k + mas[year * days + day]
    mas_year.append(k/days)
print("===============================================================================================================")
print("Yearly Mean of Average Temperature Time Series:\n")
for i in range(0, len(x_year), 9):
    print(x_year[i])
print("\nYearly Mean of Detrended Average Temperature Time Series:\n")
for i in range(0, len(mas_year), 9):
    print(mas_year[i])

# Stabilize Variance.
# Plot Yearly Variance of Average Temperature.
split = np.array_split(x, 80)
varis = []
for i in range(0, len(split)):
    varis.append(statistics.variance(split[i]))
plt.figure()
plt.plot(dates_year, varis, color='red', marker='x', linestyle='--', linewidth=1)
plt.xlabel('Time (Years)')
plt.ylabel('AvgTemp (°C)')
title = 'Yearly Variance of of Original Time Series'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)

# Print Yearly Variance of Original, Detrended and Log of Detrended Average Temperature.
split1 = np.array_split(mas, 80)
varis1 = []
for i in range(0, len(split1)):
    varis1.append(statistics.variance(split1[i]))
print("===============================================================================================================")
print("Yearly Variance of Average Temperature Time Series:\n")
for i in range(7, len(varis), 9):
    print(varis[i])
print("\nYearly Variance of Detrended Average Temperature Time Series:\n")
for i in range(7, len(varis1), 9):
    print(varis1[i])
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
split2 = np.array_split(fd, 80)
varis2 = []
for i in range(0, len(split2)):
    varis2.append(statistics.variance(split2[i]))
print("\nYearly Variance of Logarithms of Detrended Average Temperature Time Series:\n")
for i in range(7, len(varis2), 9):
    print(varis2[i])

# Print Yearly Mean Average Temperature of Logs of detrended time series.
fd_year = []
years = 80
days = 365
for year in range(0, years):
    k = 0
    for day in range(1, days + 1):
        k = k + fd[year * days + day]
    fd_year.append(k/days)
print("===============================================================================================================")
print("Yearly Mean of Logarithms of Detrended Average Temperature Time Series:\n")
for i in range(0, len(fd_year), 9):
    print(fd_year[i])

# Remove Seasonality (There is no Seasonality).

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

# Akaike Information Criterion (AIC) for choosing model order.
print("===============================================================================================================")
print("Exploring an AR Model for the Time Series:")
best_aic_ar = np.inf
best_p_ar = None
for p in np.arange(1, 10):
    try:
        _, _, _, _, aic = lf.fit_arima_model(x=fd, p=p, q=0, d=0, show=False)
    except ValueError as err:
        print("--------------------------------------------------------------------------------------------------------"
              "-------")
        print(f'AR({p}) Error ---> {err}')
        continue
    print("------------------------------------------------------------------------------------------------------------"
          "---")
    print(f'AR({p}) AIC ---> {aic}')
    if aic < best_aic_ar:
        best_p_ar = p
        best_aic_ar = aic
# best_p_ar = 3
# best_aic_ar = -2649.3628420606137
summary_ar, fittedvalues_ar, resid_ar, model_ar, aic_ar = lf.fit_arima_model(x=fd, p=best_p_ar, q=0, savepath=savepath,
                                                                             d=0, show=True)
plt.pause(0.001)
print("===============================================================================================================")
print("Summary of chosen AR Model:\n")
print(summary_ar)
nrmseV_ar, predM_ar = lf.calculate_fitting_error(fd, model_ar, best_p_ar, 0, 0, savepath=savepath, tmax=10, show=True)
plt.pause(0.001)

# Out of sample predictions for time horizon Tmax.
# Split Time Series in train and test set. Test set is the last year (365 days).
train_fd, test_fd = fd[:len(fd)-365], fd[len(fd)-365:]
model_ar_train = pm.ARIMA(order=(best_p_ar, 0, 0))
model_ar_train.fit(train_fd)
return_conf_int = True
alpha = 0.05
Tmax = round(len(test_fd)/10)
preds_ar_train, conf_bounds_ar_train = \
    lf.predict_oos_multistep(model_ar_train, tmax=Tmax, return_conf_int=return_conf_int, alpha=alpha, show=False)
plt.figure()
plt.plot(np.arange(1, Tmax+1), preds_ar_train, label='predictions')
plt.plot(np.arange(1, Tmax+1), test_fd[:Tmax], label='original')
if return_conf_int:
    plt.fill_between(np.arange(1, Tmax+1), conf_bounds_ar_train[:, 0],
                     conf_bounds_ar_train[:, 1], color='green', alpha=0.3)
plt.legend()
title = f'AR({best_p_ar}) Out of Sample Predictions for Horizon T = {Tmax}'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)
plt.pause(0.001)

# Rolling oos prediction.
preds = []
bounds = []
for i in test_fd:
    preds_ar_train_roll, conf_bounds_ar_train_roll = \
        model_ar_train.predict(n_periods=1, return_conf_int=return_conf_int, alpha=alpha)
    model_ar_train.update(i)
    preds.append(preds_ar_train_roll[0])
    bounds.append(conf_bounds_ar_train_roll[0])
plt.figure()
plt.plot(preds, label='predictions', linestyle='--', alpha=0.3)
plt.plot(test_fd, label='original', alpha=0.7)
if return_conf_int:
    bounds = np.array(bounds)
    plt.fill_between(np.arange(len(test_fd)), bounds[:, 0], bounds[:, 1], alpha=0.3, color='green')
plt.legend()
title = f'AR({best_p_ar}) Rolling Out of Sample Predictions'
plt.title(title, x=0.5, y=1.0)
plt.savefig(f'{savepath}/{title}.png')
plt.show(block=False)
plt.pause(0.001)

# Portmanteau Test to see if the residuals are white noise.
lf.portmanteau_test(resid_ar, maxtau, best_p_ar, 0, 0, savepath, show=True)

# # MA Model.
# # Autocorrelation Criterion for choosing model order.
# # Akaike Information Criterion (AIC) for choosing model order.
# print("===============================================================================================================")
# print("Exploring an MA Model for the Time Series:")
# best_aic_ma = np.inf
# best_q_ma = None
# for q in np.arange(1, 10):
#     try:
#         _, _, _, _, aic = lf.fit_arima_model(x=fd, p=0, q=q, d=0, show=False)
#     except ValueError as err:
#         print("--------------------------------------------------------------------------------------------------------"
#               "-------")
#         print(f'MA({q}) Error ---> {err}')
#         continue
#     print("------------------------------------------------------------------------------------------------------------"
#           "---")
#     print(f'MA({q}) AIC ---> {aic}')
#     if aic < best_aic_ma:
#         best_q_ma = q
#         best_aic_ma = aic
# # best_q_ma = 8
# # best_aic_ma = -2640.548
# summary_ma, fittedvalues_ma, resid_ma, model_ma, aic_ma = lf.fit_arima_model(x=fd, p=0, q=best_q_ma, savepath=savepath,
#                                                                              d=0, show=True)
# plt.pause(0.001)
# print("===============================================================================================================")
# print("Summary of chosen MA Model:\n")
# print(summary_ma)
# nrmseV_ma, predM_ma = lf.calculate_fitting_error(fd, model_ma, 0, 0, best_q_ma, savepath=savepath, tmax=10, show=True)
# plt.pause(0.001)
#
# # Out of sample predictions for time horizon Tmax.
# model_ma_train = pm.ARIMA(order=(0, 0, best_q_ma))
# model_ma_train.fit(train_fd)
# preds_ma_train, conf_bounds_ma_train = \
#     lf.predict_oos_multistep(model_ma_train, tmax=Tmax, return_conf_int=return_conf_int, alpha=alpha, show=False)
# plt.figure()
# plt.plot(np.arange(1, Tmax+1), preds_ma_train, label='predictions')
# plt.plot(np.arange(1, Tmax+1), test_fd[:Tmax], label='original')
# if return_conf_int:
#     plt.fill_between(np.arange(1, Tmax+1), conf_bounds_ma_train[:, 0],
#                      conf_bounds_ma_train[:, 1], color='green', alpha=0.3)
# plt.legend()
# title = f'MA({best_q_ma}) Out of Sample Predictions for Horizon T = {Tmax}'
# plt.title(title, x=0.5, y=1.0)
# plt.savefig(f'{savepath}/{title}.png')
# plt.show(block=False)
# plt.pause(0.001)
#
# # Rolling oos prediction.
# preds = []
# bounds = []
# for i in test_fd:
#     preds_ma_train_roll, conf_bounds_ma_train_roll = \
#         model_ma_train.predict(n_periods=1, return_conf_int=return_conf_int, alpha=alpha)
#     model_ma_train.update(i)
#     preds.append(preds_ma_train_roll[0])
#     bounds.append(conf_bounds_ma_train_roll[0])
# plt.figure()
# plt.plot(preds, label='predictions', linestyle='--', alpha=0.3)
# plt.plot(test_fd, label='original', alpha=0.7)
# if return_conf_int:
#     bounds = np.array(bounds)
#     plt.fill_between(np.arange(len(test_fd)), bounds[:, 0], bounds[:, 1], alpha=0.3, color='green')
# plt.legend()
# title = f'MA({best_q_ma}) Rolling Out of Sample Predictions'
# plt.title(title, x=0.5, y=1.0)
# plt.savefig(f'{savepath}/{title}.png')
# plt.show(block=False)
# plt.pause(0.001)
#
# # Portmanteau Test to see if the residuals are white noise.
# lf.portmanteau_test(resid_ma, maxtau, 0, 0, best_q_ma, savepath, show=True)
#
# # ARMA Model.
# # Autocorrelation Criterion for choosing model order.
# # Akaike Information Criterion (AIC) for choosing model order.
# print("===============================================================================================================")
# print("Exploring an ARMA Model for the Time Series:")
# best_aic_arma = np.inf
# best_p_arma = None
# best_q_arma = None
# for p in np.arange(1, 4):
#     for q in np.arange(1, 4):
#         try:
#             _, _, _, _, aic = lf.fit_arima_model(x=fd, p=p, q=q, d=0, show=False)
#         except ValueError as err:
#             print("----------------------------------------------------------------------------------------------------"
#                   "-----------")
#             print(f'ARMA({p},{q}) Error ---> {err}')
#             continue
#         print("--------------------------------------------------------------------------------------------------------"
#               "-------")
#         print(f'ARMA({p},{q}) AIC ---> {aic}')
#         if aic < best_aic_arma:
#             best_p_arma = q
#             best_q_arma = q
#             best_aic_arma = aic
# # best_p_arma = 3
# # best_q_arma = 5
# # best_aic_arma = np.inf
# summary_arma, fittedvalues_arma, resid_arma, model_arma, aic_arma = \
#     lf.fit_arima_model(x=fd, p=best_p_arma, q=best_q_arma, savepath=savepath, d=0, show=True)
# plt.pause(0.001)
# print("===============================================================================================================")
# print("Summary of chosen ARMA Model:\n")
# print(summary_arma)
# nrmseV_arma, predM_arma = lf.calculate_fitting_error(fd, model_arma, best_p_arma, 0, best_q_arma, savepath=savepath,
#                                                      tmax=10, show=True)
# plt.pause(0.001)
#
# # Out of sample predictions for time horizon Tmax.
# model_arma_train = pm.ARIMA(order=(best_p_arma, 0, best_q_arma))
# model_arma_train.fit(train_fd)
# preds_arma_train, conf_bounds_arma_train = \
#     lf.predict_oos_multistep(model_arma_train, tmax=Tmax, return_conf_int=return_conf_int, alpha=alpha, show=False)
# plt.figure()
# plt.plot(np.arange(1, Tmax+1), preds_arma_train, label='predictions')
# plt.plot(np.arange(1, Tmax+1), test_fd[:Tmax], label='original')
# if return_conf_int:
#     plt.fill_between(np.arange(1, Tmax+1), conf_bounds_arma_train[:, 0],
#                      conf_bounds_arma_train[:, 1], color='green', alpha=0.3)
# plt.legend()
# title = f'ARMA({best_p_arma},{best_q_arma}) Out of Sample Predictions for Horizon T = {Tmax}'
# plt.title(title, x=0.5, y=1.0)
# plt.savefig(f'{savepath}/{title}.png')
# plt.show(block=False)
# plt.pause(0.001)
#
# # Rolling oos prediction.
# preds = []
# bounds = []
# for i in test_fd:
#     preds_arma_train_roll, conf_bounds_arma_train_roll = \
#         model_arma_train.predict(n_periods=1, return_conf_int=return_conf_int, alpha=alpha)
#     model_arma_train.update(i)
#     preds.append(preds_arma_train_roll[0])
#     bounds.append(conf_bounds_arma_train_roll[0])
# plt.figure()
# plt.plot(preds, label='predictions', linestyle='--', alpha=0.3)
# plt.plot(test_fd, label='original', alpha=0.7)
# if return_conf_int:
#     bounds = np.array(bounds)
#     plt.fill_between(np.arange(len(test_fd)), bounds[:, 0], bounds[:, 1], alpha=0.3, color='green')
# plt.legend()
# title = f'ARMA({best_p_arma},{best_q_arma}) Rolling Out of Sample Predictions'
# plt.title(title, x=0.5, y=1.0)
# plt.savefig(f'{savepath}/{title}.png')
# plt.show(block=False)
# plt.pause(0.001)
#
# # Portmanteau Test to see if the residuals are white noise.
# lf.portmanteau_test(resid_arma, maxtau, best_p_arma, 0, best_q_arma, savepath, show=True)
