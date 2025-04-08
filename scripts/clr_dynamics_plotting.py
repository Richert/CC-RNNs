import numpy as np
import pickle
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
path = "/home/richard-gast/Documents"
data = pickle.load(open(f"{path}/data/clr_dynamics.pkl", "rb"))
results = read_csv(f"{path}/results/clr_dynamics.csv")

# plot raw dynamics
n_examples = 2
k = [100]
lam = [0.0]
Delta = [0.0]
sigma = [1.5]
for k_tmp, lam_tmp, Delta_tmp, sigma_tmp in zip(k, lam, Delta, sigma):

    # get index of condition
    k_idx = np.asarray(data["k"]) == k_tmp
    lam_idx = np.asarray(data["lambda"]) == lam_tmp
    delta_idx = np.asarray(data["Delta"]) == Delta_tmp
    sigma_idx = np.asarray(data["sigma"]) == sigma_tmp
    final_idx = 1.0 * k_idx * lam_idx * delta_idx * sigma_idx

    # create figure
    fig = plt.figure(figsize=(12, 3*n_examples))
    grid = fig.add_gridspec(nrows=n_examples, ncols=2)
    for idx in range(n_examples):
        trial = np.random.choice(np.argwhere(final_idx > 0).squeeze())
        ax = fig.add_subplot(grid[idx, 0])
        im = ax.imshow(data["z"][trial].T, aspect="auto", interpolation="none", cmap="cividis")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("steps")
        ax.set_ylabel("neurons")
        ax.set_title(f"trial {trial + 1} - unperturbed dynamics")
        ax = fig.add_subplot(grid[idx, 1])
        im = ax.imshow(data["z_p"][trial].T, aspect="auto", interpolation="none", cmap="cividis")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("steps")
        ax.set_ylabel("neurons")
        ax.set_title(f"trial {trial + 1} - perturbed dynamics")
    plt.tight_layout()

# plot lyapunov exponents
k = [100]
lam = [0.0]
for k_tmp, lam_tmp in zip(k, lam):

    # get index of condition
    k_idx = np.asarray(data["k"]) == k_tmp
    lam_idx = np.asarray(data["lambda"]) == lam_tmp
    final_idx = 1.0 * k_idx * lam_idx

    # create 2D matrix of LE
    df = results.loc[final_idx > 0, :]
    deltas = np.unique(df.loc[:, "Delta"].values)
    sigmas = np.unique(df.loc[:, "sigma"].values)
    LE = np.zeros((len(deltas), len(sigmas)))
    for i, Delta in enumerate(deltas):
        for j, sigma in enumerate(sigmas):
            idx1 = df.loc[:, "Delta"].values == Delta
            idx2 = df.loc[:, "sigma"].values == sigma
            LE[i, j] = np.mean(df.loc[idx1 & idx2, "lyapunov"].values)
    LE = DataFrame(columns=sigmas, index=deltas, data=LE)

    # create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sb.heatmap(LE, ax=ax, cmap="cividis", vmin=-0.3, vmax=0.3)
    ax.set_xlabel("sigma")
    ax.set_ylabel("Delta")
    plt.tight_layout()

plt.show()
