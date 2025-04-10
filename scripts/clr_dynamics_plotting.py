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
n_examples = 1
lam = [0.0]
Delta = [0.0]
sigma = [1.5]
for lam_tmp, Delta_tmp, sigma_tmp in zip(lam, Delta, sigma):

    # get index of condition
    lam_idx = np.asarray(data["lambda"]) == lam_tmp
    delta_idx = np.asarray(data["Delta"]) == Delta_tmp
    sigma_idx = np.asarray(data["sigma"]) == sigma_tmp
    final_idx = 1.0 * lam_idx * delta_idx * sigma_idx

    # create figure
    fig = plt.figure(figsize=(12, 5*n_examples))
    grid = fig.add_gridspec(nrows=n_examples*3, ncols=1)
    for idx in range(n_examples):
        trial = np.random.choice(np.argwhere(final_idx > 0).squeeze())
        ax = fig.add_subplot(grid[idx*3, 0])
        im = ax.imshow(data["z_unperturbed"][trial].T, aspect="auto", interpolation="none", cmap="cividis")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("steps")
        ax.set_ylabel("neurons")
        ax.set_title(f"trial {trial + 1} - unperturbed dynamics")
        ax = fig.add_subplot(grid[idx*3+1, 0])
        im = ax.imshow(data["z_perturbed"][trial].T, aspect="auto", interpolation="none", cmap="cividis")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("steps")
        ax.set_ylabel("neurons")
        ax.set_title(f"trial {trial + 1} - perturbed dynamics")
        ax = fig.add_subplot(grid[idx * 3 + 2, 0])
        im = ax.imshow(data["z_memory"][trial].T, aspect="auto", interpolation="none", cmap="cividis")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("steps")
        ax.set_ylabel("neurons")
        ax.set_title(f"trial {trial + 1} - random dynamics")
    plt.tight_layout()

# plot 2D matrices
lam = [0.0]
for lam_tmp in lam:

    # reduce data to the plotting selection
    lam_idx = np.asarray(data["lambda"]) == lam_tmp
    final_idx = 1.0 * lam_idx
    df = results.loc[final_idx > 0, :]
    deltas = np.unique(df.loc[:, "Delta"].values)
    sigmas = np.unique(df.loc[:, "sigma"].values)

    # create 2D matrix of LE, MC, and TS
    LE = np.zeros((len(deltas), len(sigmas)))
    C = np.zeros_like(LE)
    TS = np.zeros_like(LE)
    for i, Delta in enumerate(deltas):
        for j, sigma in enumerate(sigmas):
            idx1 = df.loc[:, "Delta"].values == Delta
            idx2 = df.loc[:, "sigma"].values == sigma
            LE[i, j] = np.mean(df.loc[idx1 & idx2, "lyapunov"].values)
            C[i, j] = np.mean(df.loc[idx1 & idx2, "memory"].values)
            TS[i, j] = np.mean(df.loc[idx1 & idx2, "timescale_heterogeneity"].values)
    LE = DataFrame(columns=sigmas, index=deltas, data=LE)
    C = DataFrame(columns=sigmas, index=deltas, data=C)
    TS = DataFrame(columns=sigmas, index=deltas, data=TS)

    # create LE figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sb.heatmap(LE, ax=ax, cmap="vlag", vmin=-0.3, vmax=0.3)
    ax.set_xlabel("sigma")
    ax.set_ylabel("Delta")
    ax.set_title("Maximum Lyapunov Exponent")
    plt.tight_layout()

    # create MC figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sb.heatmap(C, ax=ax, cmap="cividis")
    ax.set_xlabel("sigma")
    ax.set_ylabel("Delta")
    ax.set_title("Memory Capacity")
    plt.tight_layout()

    # create TS figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sb.heatmap(TS, ax=ax, cmap="cividis")
    ax.set_xlabel("sigma")
    ax.set_ylabel("Delta")
    ax.set_title("Timescale Heterogeneity")
    plt.tight_layout()

plt.show()
