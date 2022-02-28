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


# Theoretical autocorrelation function of an ARMA process. phiV: The coefficients for autoregressive lag
# polynomial, not including zero lag. thetaV : array_like, 1d. The coefficients for moving-average lag polynomial, not
# including zero lag.
def armacoefs2autocorr(phiv, thetav, lags=10):
    phiv, thetav = np.array(phiv), np.array(thetav)
    phiv = np.r_[1, -phiv]  # add zero lag
    thetav = np.r_[1, thetav]  # #add zero lag
    acf_ = arma_acf(phiv, thetav, lags=lags)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.scatter(np.arange(1, lags), acf_[1:], marker='o')
    ax.set_xlabel('Lags')
    ax.set_xticks(np.arange(1, lags))
    ax.set_yticks(np.arange(-1, 1, 0.1))
    ax.set_title('ACF', fontsize=14)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
    plt.show()
    # for t in np.arange(lags):
    #     ax.axvline(t, ymax=acf_[t], color='red', alpha=0.3)


def macoef2autocorr(phiv, thetav, lags=10):
    pacf_ = arma_pacf(phiv, thetav, lags=10)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(lags), pacf_, marker='o')
    for t in np.arange(lags):
        ax.axvline(t, ymax=pacf_[t], color='red', alpha=0.3)


# Plot 3d attractor.
def plot_3d_attractor(xm, savepath, title):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xm[:, [0]], xm[:, [1]], xm[:, [2]])
    plt.title(title)
    plt.savefig(f'{savepath}/{title}.png')
    plt.show()


# Time-delay embedding. x : 1d-array, shape (n_times) Time series. m : int Embedding dimension (order).
# tau : int Delay. Returns embedded : ndarray, shape (n_times - (order - 1) * delay, order) Embedded time-series.
def embed_data(xv, m=3, tau=1):
    n = len(xv)
    nvec = n - (m - 1) * tau
    xm = np.zeros(shape=(nvec, m))
    for i in np.arange(m):
        xm[:, m - i - 1] = xv[i * tau:nvec + i * tau]
    return xm


# Start prediction after getting all lags needed from model.
def predict_multistep(model, tmax=10, show=False):
    tmin = np.max([len(model.arparams), len(model.maparams), 1])
    preds = model.predict(start=tmin, end=tmax, dynamic=True)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(preds)
        ax.set_title('Multistep prediction')
        ax.set_xlabel('T')
        plt.show()
    return preds


# Transform a variable of any distribution into normal.
def gaussianisation(data):
    sort_ind = np.argsort(data)
    gaussian_data = np.random.normal(0, 1, size=data.shape[0])
    gaussian_data_ind = np.argsort(gaussian_data)
    g_d_sorted = gaussian_data[gaussian_data_ind]
    y = np.zeros(shape=data.shape[0])
    for i in np.arange(data.shape[0]):
        y[i] = g_d_sorted[sort_ind[i]]
    return y


# Get NRMSE.
def get_nrmse(target, predicted):
    se = (target - predicted) ** 2
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse / np.std(target)


# NRMSE Definition.
def nrmse(truev, predictedv):
    n = mean_squared_error(truev, predictedv, squared=False)/(max(truev)-min(truev))
    return n


# LINEARFITNRMSE fits an AR model and computes the fitting error for t-step ahead. INPUTS: xV: vector of the scalar
# time series. m: the embedding dimension. Tmax    : the prediction horizon, the fit is made for t=1...Tmax steps ahead.
# tittxt: string to be displayed in the title of the figure if not specified, no plot is made. OUTPUT: nrmsev: vector
# of length Tmax, the nrmse of the fit for t-mappings, t=1...Tmax. phiv: the coefficients of the estimated AR time
# series of length (m+1) with phi(0) as first component.
def linearfitnrmse(xv, m, tmax=1, show=False):
    n = xv.shape[0]
    mx = np.mean(xv[:n - tmax + 1])
    yv = xv[:n - tmax + 1] - mx
    nvec = n - m - 1 - tmax
    ym = np.full(shape=(nvec-1, m), fill_value=np.nan)
    for j in np.arange(m):
        ym[:, [m-j-1]] = yv[j:nvec+j-1]
    rv = yv[m:nvec+m-1]
    # np.linalg.lstsq(ym, rv)
    ols = OLS(endog=rv, exog=ym).fit()
    av = ols.params
    a0 = (1 - np.sum(av)) * mx
    phiv = np.r_[a0, av]
    prem = np.full(shape=(n + tmax - 1, tmax), fill_value=np.nan)
    for i in np.arange(m, n):
        prev = np.full(shape=(m + tmax, 1), fill_value=np.nan)
        prev[:m] = xv[i - m: i] - mx
        for T in np.arange(1, tmax + 1):
            prev[m + T - 1] = np.dot(av, (prev[T-1:m+T-1][::-1]))
            prem[i + T - 1, [T-1]] = prev[m + T - 1]
    prem = prem + mx
    nrmsev = np.ones(shape=(tmax, 1))
    for T in np.arange(1, tmax + 1):
        nrmsev[T-1] = nrmse(xv[m + T - 1:n], prem[m + T - 1: n, [T - 1]])
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, tmax + 1), nrmsev, marker='x')
        ax.set_xlabel('prediction time T')
        ax.set_ylabel('NRMSE(T)')
    return nrmsev, phiv


# LOCALFITNRMSE makes fitting using a local model of zeroth order (average mapping or nearest neighbor mappings if only
# one neighbor is chosen) or a local linear model and computes the fitting error for t-step ahead. For the search for
# neighboring points it uses the Matlab k-d-tree search. The fitting here means that predictions are made for all the
# points in the data set (in-sample prediction). The prediction error statistic (NRMSE measure) for the t-step ahead
# predictions is the goodness-of-fit statistic. The state space reconstruction is done with the method of delays having
# as parameters the embedding dimension 'm' and the delay time 'tau'. The local prediction model is one of the
# following: Ordinary Least Squares, OLS (standard local linear model): if the truncation parameter q >= m. Principal
# Component Regression, PCR, project the parameter space of the model to only q of the m principal axes: if 0<q<m. Local
# Average Mapping, LAM: if q=0. The local region is determined by the number of neighbours 'nnei'. The k-d-tree data
# structure is utilized to speed up computation time in the search of neighboring points and the implementation of
# Matlab is used. INPUTS: xV: vector of the scalar time series. tau: the delay time (usually set to 1). m: the embedding
# dimension. Tmax: the prediction horizon, the fit is made for t=1...Tmax steps ahead. nnei: number of nearest neighbors
# to be used in the local model. If k=1,the nearest neighbor mapping is the fitted value. If k>1, the model as defined
# by the input patameter 'q' is used. q: the truncation parameter for a normalization of the local linear model if
# specified (to project the parameter space of the model, using Principal Component Regression, PCR, locally). if
# q>=m -> Ordinary Least Squares, OLS (standard local linear model, no projection). if 0<q<m -> PCR(q). if q=0 ->
# local average model (if in addition nnei=1 -> then the zeroth order model is applied). tittxt  : string to be
# displayed in the title of the figure if not specified, no plot is made. OUTPUT: nrmsev: vector of length Tmax, the
# nrmse of the fit for t-mappings, t=1...Tmax. prem: the matrix of size nvec x (1+Tmax) having the fit (in-sample
# predictions) for t=1,...,Tmax, for each of the nvec reconstructed points from the whole time series. The first column
# has the time of the target point and the rest Tmax columns the fits for t=1,...,Tmax time steps ahead.
def localfitnrmse(xv, tau, m, tmax, nnei, q, show=''):
    if q > m:
        q = int(m)
    n = xv.shape[0]
    if n < 2 * (m-1)*tau - tmax:
        print('too short timeseries')
        return
    nvec = n - (m-1) * tau - tmax
    xm = np.full(shape=(nvec, m), fill_value=np.nan)
    for j in np.arange(m):
        xm[:, [m-j-1]] = xv[j * tau:nvec + j * tau]
    from scipy.spatial import KDTree
    kdtrees = KDTree(xm)
    prem = np.full(shape=(nvec, tmax), fill_value=np.nan)
    _, nneiindm = kdtrees.query(xm, k=nnei+1, p=2)
    nneiindm = nneiindm[:, 1:]
    for i in np.arange(nvec):
        neim = xm[nneiindm[i]]
        yv = xv[nneiindm[i] + m * tau]
        if q == 0 or nnei == 1:
            prem[i, 0] = np.mean(yv)
        else:
            mneiv = np.mean(neim, axis=0)
            my = np.mean(yv)
            zm = neim - mneiv
            [ux, sx, vx] = np.linalg.svd(zm, full_matrices=False)
            sx = np.diag(sx)
            vx = vx.T
            tmpm = vx[:, :q] @ (np.linalg.inv(sx[:q, :q]) @ ux[:, :q].T)
            lsbv = tmpm @ (yv - my)
            prem[i] = my + (xm[i, ] - mneiv) @ lsbv
    if tmax > 1:
        winnowm = np.full(shape=(nvec, (m - 1) * tau + 1), fill_value=np.nan)
        for i in np.arange(m*tau):
            winnowm[:, [i]] = xv[i:nvec + i]
        for t in np.arange(2, tmax + 1):
            winnowm = np.concatenate([winnowm, prem[:, [t-2]]], axis=1)
            targm = winnowm[:, :-(m+1)*tau:-tau]
            _, nneiindm = kdtrees.query(targm, k=nnei, p=2)

            for i in np.arange(nvec):
                neim = xm[nneiindm[i], :]
                yv = xv[nneiindm[i] + (m - 1) * tau + 1]
                if q == 0 or nnei == 1:
                    prem[i, t-1] = np.mean(yv)
                else:
                    mneiv = np.mean(neim, axis=0)
                    my = np.mean(yv)
                    zm = neim - mneiv
                    [ux, sx, vx] = np.linalg.svd(zm, full_matrices=False)
                    sx = np.diag(sx)
                    vx = vx.T
                    tmpm = vx[:, :q] @ (np.linalg.inv(sx[:q, :q]) @ ux[:, :q].T)
                    lsbv = tmpm @ (yv - my)
                    prem[i, t-1] = my + (targm[i, :] - mneiv) @ lsbv
    nrmsev = np.full(shape=(tmax, 1), fill_value=np.nan)
    idx = (np.arange(nvec) + (m-1)*tau).astype(np.int)
    for t_idx in np.arange(1, tmax + 1):
        nrmsev[t_idx-1] = nrmse(truev=xv[idx + t_idx, ], predictedv=prem[:, [t_idx - 1]])
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, tmax + 1), nrmsev, marker='x')
        ax.set_xlabel('prediction time t')
        ax.set_ylabel('NRMSE(t)')
    return nrmsev, prem


# Helper Function.
def ann(x, k):
    tree = KDTree(x, leaf_size=1, metric='chebyshev')
    dists, nnidx = tree.query(x, k=k)
    del tree
    return nnidx, dists


# Helper Function.
def annr(x, rv):
    tree = KDTree(x, leaf_size=1, metric='chebyshev')
    nnnidx = tree.query_radius(x, r=rv, count_only=True)
    return nnnidx


# Helper Function.
def nneighforgivenr(x, rv):
    npv = annr(x, rv)
    npv[npv == 0] = 1
    return npv


# Calculates I(X;Y) using KSG algorithm1 (with max-norms squares).
def mi_estimator_ksg1(xv, yv, nnei=5, normalize=False):
    n = xv.shape[0]
    psi_nnei = psi(nnei)
    psi_n = psi(n)
    if normalize:
        xv = (xv - np.min(xv)) / np.ptp(xv)
        yv = (yv - np.min(yv)) / np.ptp(yv)
    xembm = np.concatenate((xv, yv), axis=1)
    _, distsm = ann(xembm, nnei + 1)
    maxdistv = distsm[:, -1]
    n_x = nneighforgivenr(x=xv, rv=maxdistv - np.ones(n) * 10 ** (-10))
    n_y = nneighforgivenr(x=yv, rv=maxdistv - np.ones(n) * 10 ** (-10))
    psibothm = psi(np.concatenate((n_x.reshape(-1, 1), n_y.reshape(-1, 1)), axis=1))
    # I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Ny + 1)>
    mi = psi_nnei + psi_n - np.mean(np.sum(psibothm, axis=1))
    return mi


def falsenearestneighbors(xv, m_max=10, tau=1, show=False):
    dim = np.arange(1, m_max + 1)
    f1, _, _ = dimension.fnn(xv, tau=tau, dim=dim, window=10, metric='cityblock', parallel=False)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.scatter(dim, f1)
        ax.axhline(0.01, linestyle='--', color='red', label='1% threshold')
        ax.set_xlabel(f'm')
        ax.set_title(f'FNN ({m_max})')
        ax.set_xticks(dim)
        ax.legend()
        plt.show()
    return f1


def correlationdimension(xv, mmax, show=False):
    m_all = np.arange(1, mmax + 1)
    corrdimv = []
    logrm = []
    logcrm = []
    polym = []
    for m in m_all:
        corrdim, *corrData = nolds.corr_dim(xv, m, debug_data=True)
        corrdimv.append(corrdim)
        logrm.append(corrData[0][0])
        logcrm.append(corrData[0][1])
        polym.append(corrData[0][2])
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(m_all, corrdimv, marker='x', linestyle='.-')
        ax.set_xlabel('m')
        ax.set_xticks(m_all)
        ax.set_ylabel('v')
        ax.set_title('Corr Dim vs m')
    return corrdimv, logrm, logcrm, polym


def split2train_testset(xv, test_proportion):
    n = np.int(len(xv) * test_proportion)
    return xv[:n], xv[n:]


def logisticmap(n=1024, r=3., x0=None):
    ntrans = 10
    xv = np.full(shape=(n+ntrans, 1), fill_value=np.nan)
    if x0 is None:
        xv[0] = np.random.uniform(low=0, high=2.9)
    else:
        xv[0] = x0
    for t in np.arange(1, n+ntrans):
        xv[t] = r * xv[t-1] * (1 - xv[t-1])
    xv = xv[ntrans:, [0]]
    return xv.reshape(-1, )

