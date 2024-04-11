"""
Helper functions class
"""

import torch
import scipy.sparse as sparse
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np


def conc_sv(C):
    SVS = []
    for i, c in zip(range(len(C)),C):
        SVS.append(np.linalg.svd(c, full_matrices=True)[1])
    return SVS


def checkRecall(patterns, Y_recalls, evalRange = 50):
    """
    :Description: Function that calculates the mean error between a target pattern and a recall from a RFC
    
    :Parameters:
        1. patterns:    list with entries for each target pattern. Each entry should be 2-dimensional
                        with time on the first dimension and the features on the second dimension
        2. Y_recalls:   list with same number of entries as patterns, whose dimensionality should match
                        the entries in patterns as well. Recalls do not have to be of same time length as patterns
        3. evalRange:   time range over which to evaluate the mean error. If None, length of Y_recall will be taken for each pattern (default = 50)
    
    :Returns:
        1. meanError:   mean missmatch between patterns and Y_recalls. 0 = no missmatch, 1 = complete missmatch
    """
    
    meanError = np.zeros([len(patterns)])
    
    # loop over patterns
    for i,p in enumerate(patterns):
        
        if evalRange is None: evalRange = len(Y_recalls[i])
        target = np.argmax(p[0:evalRange,:], axis = 1)
        recall = np.argmax(Y_recalls[i], axis = 1)
        
        # calculate phasematch between target and recall for each phaseshift
        L = len(recall)
        M = len(target)
        phasematches = np.zeros([L-M])
        for s in range(L-M):
            phasematches[s] = np.linalg.norm(target-recall[s:s+M])
        
        # use position of maximal phasematch to calculate mean error
        pos = np.argmin(phasematches)
        recall_pm = recall[pos:pos+evalRange]
        target_pm = target[0:evalRange]
        
        meanError[i] = np.mean(recall_pm != target_pm)
    
    return meanError


def init_weights(N: int, M: int, density: float):
    W_raw = sparse.rand(N, M, format='lil', density=density)
    W = np.zeros((N, M))
    rows, cols = W_raw.nonzero()
    for row, col in zip(rows, cols):
        W[row, col] = np.random.randn()
    try:
        lambdas = np.abs(np.linalg.eigvals(W))
        W = np.squeeze(np.asarray(W/np.max(lambdas)))
    except np.linalg.LinAlgError:
        pass
    return W


def nrmse(y: np.ndarray, target: np.ndarray):
    combinedVar = 0.5 * (np.var(target,1) + np.var(y,1))
    error = y-target
    return np.sqrt(np.mean(error**2, axis=1)/combinedVar)


def tensor_nrmse(y: torch.tensor, target: torch.tensor) -> torch.tensor:
    combined_var = 0.5 * (torch.var(target, 1) + torch.var(y, 1))
    error = y - target
    return torch.sqrt(torch.mean(error ** 2, dim=1) / combined_var)


def ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    I = np.eye(X.shape[0])
    return np.transpose(np.dot(np.linalg.pinv(np.dot(X, X.T) + alpha*I), np.dot(X, y.T)))


def tensor_ridge(X: torch.tensor, Y: torch.tensor, alpha: float) -> torch.tensor:
    I = torch.eye(X.shape[0])
    return (torch.linalg.pinv(X @ X.T + alpha*I) @ X @ Y.T).T


def AND(C,B):
    dim = len(C)
    tol = 1e-12
    
    Uc,Sc,Vc = np.linalg.svd(C, full_matrices=True)      
    Ub,Sb,Vb = np.linalg.svd(B, full_matrices=True) 

    if np.diag(Sc[Sc > tol]).size:
        numRankC = np.linalg.matrix_rank(np.diag(Sc[Sc > tol]))
    else:
        numRankC = 0
    if np.diag(Sb[Sb > tol]).size:    
        numRankB = np.linalg.matrix_rank(np.diag(Sb[Sb > tol]))  
    else:
        numRankB = 0 
        
    Uc0 = Uc[:, numRankC:]
    Ub0 = Uc[:, numRankB:]
    
    W,Sig,Wt = np.linalg.svd(np.dot(Uc0,Uc0.T)+np.dot(Ub0,Ub0.T), full_matrices=True)      
    if np.diag(Sig[Sig > tol]).size: 
        numRankSig = np.linalg.matrix_rank(np.diag(Sig[Sig > tol]))
    else: 
        numRankSig = 0
    Wgk = W[:, numRankSig:]
    arg = np.linalg.pinv(C,tol)+np.linalg.pinv(B,tol)-np.eye(dim)
    
    return np.dot(np.dot(Wgk,np.linalg.inv(np.dot(Wgk.T,np.dot(arg,Wgk)))),Wgk.T)


def NOT(C):
    dim = len(C)
    return np.eye(dim) - C


def OR(C,B):
    return NOT(AND(NOT(C), NOT(B)))


def sPHI(c,gamma):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if (gamma == 0):
            if (c[i] < 1):  d[i] = 0
            if (c[i] == 1): d[i] = 1
        else:
            d[i] = c[i]/(c[i]+(gamma**-2)*(1-c[i]))
    return d


def sNOT(c):
    return 1-c


def sAND(c,b):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if (c[i] == 0 and b[i] == 0):
            d[i] = 0
        else:
            d[i] = c[i]*b[i]/(c[i]+b[i]-c[i]*b[i])
    return d


def sOR(c,b):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if (c[i] == 1 and c[i] == b[i]):
            d[i] = 1
        else:
            d[i] = (c[i]+b[i]-2*c[i]*b[i])/(1-c[i]*b[i])
    return d


def phi(C, gamma):
    return np.dot(C, np.linalg.inv(C+gamma**(-2)*(np.eye(len(C))-C)))


def plot_interpolate_1d(patterns, Y_recalls, overSFac=20, plotrange=30):
    Driver_int = np.zeros([(plotrange - 1) * overSFac])
    Recall_int = np.zeros([(len(Y_recalls[0]) - 1) * overSFac])
    NRMSEsAlign = np.zeros([len(patterns)])

    for i, p in zip(range(len(patterns)), patterns):

        p = np.vectorize(p)

        Driver = p(np.linspace(0, plotrange - 1, plotrange))
        Recall = np.squeeze(Y_recalls[i])

        fD = interpolate.interp1d(range(plotrange), Driver, kind='cubic')
        fR = interpolate.interp1d(range(len(Recall)), Recall, kind='cubic')

        Driver_int = fD(np.linspace(0, (len(Driver_int) - 1.) / overSFac, len(Driver_int)))
        Recall_int = fR(np.linspace(0, (len(Recall_int) - 1.) / overSFac, len(Recall_int)))

        L = len(Recall_int)
        M = len(Driver_int)

        phasematches = np.zeros([L - M])

        for s in range(L - M):
            phasematches[s] = np.linalg.norm(Driver_int - Recall_int[s:s + M])

        pos = np.argmin(phasematches)
        Recall_PL = Recall_int[np.linspace(pos, pos + overSFac * (plotrange - 1), plotrange).astype(int)]
        Driver_PL = Driver_int[np.linspace(0, overSFac * (plotrange - 1) - 1, plotrange).astype(int)]

        NRMSEsAlign[i] = nrmse(np.reshape(Recall_PL, (1, len(Recall_PL))), np.reshape(Driver_PL, (1, len(Driver_PL))))

        plt.subplot(len(patterns), 1, i + 1)
        xspace = np.linspace(0, plotrange - 1, plotrange)
        plt.plot(xspace, Driver_PL)
        plt.plot(xspace, Recall_PL)

    print(NRMSEsAlign)
