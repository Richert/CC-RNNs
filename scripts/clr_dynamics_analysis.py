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


def timescale_heterogeneity(x: np.ndarray) -> float:
    fourier_transforms = []
    for i in range(x.shape[1]):
        x_ft = np.abs(np.fft.rfft(x[:, i] - np.mean(x[:, i])))
        fourier_transforms.append(x_ft)
    z = np.sum(fourier_transforms, axis=0)
    H = entropy(z / np.sum(z)) * np.sum(z)

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

    return H


# load data
path = "/home/richard-gast/Documents/"
load_file = f"{path}/data/clr_dynamics.pkl"
save_file = f"{path}/results/clr_dynamics.csv"
data = pickle.load(open(load_file, "rb"))

# prepare data frame
lyapunov = np.zeros((len(data["trial"]),))
memory = np.zeros((len(data["trial"])))
columns = list(data.keys())
for key in ["z_perturbed", "z_unperturbed", "z_memory", "x"]:
    columns.pop(columns.index(key))
df = DataFrame(columns=columns + ["lyapunov", "memory", "timescale_heterogeneity"], index=np.arange(0, len(lyapunov)))

# analysis of model dynamics
d_max = 20
for n in range(len(lyapunov)):

    # calculate maximum lyapunov exponent
    z, z_p = data["z_unperturbed"][n], data["z_perturbed"][n]
    d0 = np.sqrt(np.sum((z[0] - z_p[0])**2))
    d1 = np.sqrt(np.sum((z[-1] - z_p[-1])**2))
    le = np.log(d1/d0) / len(z)

    # calculate memory capacity
    x, z = data["x"][n], data["z_memory"][n]
    mc = memory_capacity(x, z, d_max, alpha=1e-4)

    # calculate time scale heterogeneity
    z = data["z_unperturbed"][n]
    ts = timescale_heterogeneity(z)

    # store results
    for c in columns:
        df.loc[n, c] = data[c][n]
    df.loc[n, "lyapunov"] = le
    df.loc[n, "memory"] = np.sum(mc)
    df.loc[n, "timescale_heterogeneity"] = ts

# save results
df.to_csv(save_file)
