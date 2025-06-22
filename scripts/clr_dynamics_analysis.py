import sys
sys.path.append('../')
import numpy as np
import pickle
from pandas import DataFrame
from src.functions import ridge
import matplotlib.pyplot as plt


def entropy(x: np.ndarray) -> float:
    y_col = []
    for i in range(x.shape[0]):
        y = x[i] * np.log(x[i]) if x[i] > 0 else 0.0
        y_col.append(y)
    return -np.sum(y_col)


def memory_capacity(x: np.ndarray, y: np.ndarray, d_max: int, alpha: float = 1e-4) -> list:
    capacities = []
    for d in range(1, d_max+1):
        x_tmp, y_tmp = x[:-d].T, y[d:].T
        W_out = ridge(y_tmp, x_tmp, alpha)
        x_pred = W_out @ y_tmp
        capacities.append(np.corrcoef(x_tmp, x_pred)[0, 1]**2)
    return capacities


def participation_ratio(x: np.ndarray):
    x_norm = np.asarray([x[:, i] - np.mean(x[:, i]) for i in range(x.shape[1])])
    cov = x_norm @ x_norm.T
    lambdas = np.linalg.eigvals(cov)
    l_real = np.real(lambdas)
    return np.sum(l_real)**2/np.sum(l_real**2)/cov.shape[0]


def timescale_heterogeneity(x: np.ndarray, normalize: bool = False) -> tuple:
    fourier_transforms = []
    for i in range(x.shape[1]):
        x_max = np.max(np.abs(x[:, i]))
        if x_max > 0.1 and normalize:
            x_norm = x[:, i] / x_max
        else:
            x_norm = x[:, i]
        x_ft = np.abs(np.fft.rfft(x_norm))
        fourier_transforms.append(x_ft)
    z = np.sum(fourier_transforms, axis=0)
    H = entropy(z / np.sum(z))

    # fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
    # ax = axes[0]
    # im = ax.imshow(x.T, aspect="auto", interpolation="none")
    # plt.colorbar(im, ax=ax)
    # ax.set_xlabel("steps")
    # ax.set_ylabel("neurons")
    # ax.set_title("Raw signals")
    # ax = axes[1]
    # im = ax.imshow(np.asarray(fourier_transforms), aspect="auto", interpolation="none")
    # plt.colorbar(im, ax=ax)
    # ax.set_xlabel("freqs")
    # ax.set_ylabel("neurons")
    # ax.set_title("FFT signals")
    # ax = axes[2]
    # ax.bar(np.arange(len(z)), z, width=0.7)
    # ax.set_xlabel("freqs")
    # ax.set_ylabel("p")
    # ax.set_title(f"timescale distribution: H = {np.round(H, decimals=3)}")
    # plt.tight_layout()
    # plt.show()

    return np.sqrt(H*np.sum(z)/len(z)), np.sqrt(np.sum(z)**3/np.sum(z**2)/len(z))


# load data
path = "/home/richard"
load_file = f"{path}/data/dendritic_gain_dynamics.pkl"
save_file = f"{path}/results/dendritic_gain_dynamics.csv"
data = pickle.load(open(load_file, "rb"))

# prepare data frame
lyapunov = np.zeros((len(data["trial"]),))
memory = np.zeros((len(data["trial"])))
columns = list(data.keys())
n_trials = len(lyapunov)
measures = ["lyapunov", "memory", "entropy", "psd", "dimensionality"]
for key in ["z_noinp", "z_inp", "z_inp_p", "x"]:
    columns.pop(columns.index(key))
df = DataFrame(columns=columns + measures, index=np.arange(0, len(lyapunov)))

# analysis of model dynamics
d_max = 40
alpha = 1e-4
for n in range(n_trials):

    # calculate maximum lyapunov exponent
    z, z_p = data["z_inp"][n], data["z_inp_p"][n]
    d0 = np.sqrt(np.sum((z[0] - z_p[0])**2))
    d1 = np.sqrt(np.sum((z[-1] - z_p[-1])**2))
    le = np.log(d1/d0) / len(z)

    # calculate memory capacity
    x, z = data["x"][n], data["z_inp"][n]
    mc = memory_capacity(x, z, d_max, alpha=alpha)

    # calculate time scale heterogeneity
    z = data["z_noinp"][n]
    H, psd = timescale_heterogeneity(z)

    # calculate dimensionality
    z = data["z_noinp"][n]
    dim = participation_ratio(z)

    # store results
    for c in columns:
        df.loc[n, c] = data[c][n]
    df.loc[n, "lyapunov"] = le
    df.loc[n, "memory"] = np.sum(mc)
    df.loc[n, "entropy"] = H
    df.loc[n, "psd"] = psd
    df.loc[n, "dimensionality"] = dim

    print(f"Finished trial {n+1} out of {n_trials}")

# save results
df.to_csv(save_file)
