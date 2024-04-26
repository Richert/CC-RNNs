import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from scipy.stats import wasserstein_distance
import os
import pandas as pd


def wasserstein(x: np.ndarray, y: np.ndarray, n_bins: int = 100) -> tuple:

    # get histograms of arrays
    x_hist, bin_edges = np.histogram(x, bins=n_bins, density=True)
    y_hist, _ = np.histogram(y, bins=bin_edges, density=True)
    x_hist /= np.sum(x_hist)
    y_hist /= np.sum(y_hist)

    # calculate KLD
    wd = wasserstein_distance(x_hist, y_hist)
    return wd, x_hist, y_hist, bin_edges


# data collection and analysis
##############################

files = os.listdir("../results/rfc_lorenz")
df = pd.DataFrame(columns=["alpha", "rep", "wd", "k", "dim"], index=np.arange(0, len(files)), dtype=np.float64)
n_bins = 500
eps = 1e-10
for n, file in enumerate(files):

    # load data
    data = pickle.load(open(f"../results/rfc_lorenz/{file}", "rb"))
    alpha = data["condition"]["alpha"]
    rep = data["condition"]["repetition"]

    # calculate dimensionality
    k_star = np.sum(data["c"])
    dim = np.sum(data["c"] > eps)

    # calculate wasserstein distance
    wd = 0.0
    for i in range(data["targets"].shape[1]):
        targets = data["targets"][:, i]
        predictions = data["predictions"][:, i]
        wd_tmp, *_ = wasserstein(targets, predictions, n_bins=n_bins)
        wd += wd_tmp

    df.loc[n, :] = (alpha, rep, wd, k_star, dim)

# collect representative trajectories for target alphas
alphas = [2.0, 4.0, 8.0]
trajectories = []
for alpha in alphas:

    # find representative sample
    idx = np.round(df.loc[:, "alpha"].values, decimals=1) == alpha
    wd_mean = np.mean(df.loc[idx, "wd"].values)
    idx = np.argmin(np.abs(df.loc[:, "wd"].values - wd_mean))
    rep = df.at[idx, "rep"]

    # load data
    data = pickle.load(open(f"../results/rfc_lorenz/alpha{int(alpha)}_{int(rep)}.pkl", "rb"))
    trajectories.append(data["predictions"])

# get target trajectory
targets = pickle.load(open(f"../results/rfc_lorenz/{files[0]}", "rb"))["targets"]

# plotting
##########

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 8
plot_start = 2000
plot_stop = 7000

# figure layout
fig = plt.figure(figsize=(12, 8))
subfigs = fig.subfigures(nrows=2)

# violin plots
x = "alpha"
ys = ["wd", "k", "dim"]
titles = ["Wasserstein Distance", r"$k^*$", "d"]
axes = subfigs[0].subplots(ncols=3)
for i in range(len(ys)):

    ax = axes[i]
    sb.violinplot(x=x, y=ys[i], data=df, color="0.8", ax=ax)
    sb.stripplot(x=x, y=ys[i], data=df, jitter=True, zorder=1, ax=ax)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(titles[i])

subfigs[0].suptitle("RFC reconstruction quality and dimensionality")

# trajectory plots
data = trajectories + [targets]
titles = [fr"$\alpha = {alpha}$" for alpha in alphas] + ["Lorenz equations"]
colors = ["darkgreen", "darkblue", "darkred", "black"]
lvars = [0, 2]
axes = subfigs[1].subplots(ncols=len(titles))
for i in range(len(titles)):

    ax = axes[i]
    ax.plot(data[i][plot_start:plot_stop, lvars[0]], data[i][plot_start:plot_stop, lvars[1]], color=colors[i])
    ax.set_xlabel(fr"$u_{lvars[0] + 1}$")
    ax.set_ylabel(fr"$u_{lvars[1] + 1}$")
    ax.set_title(titles[i])

subfigs[1].suptitle("Examples of RFC reconstructions")

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.02)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'../results/rfc_lorenz.svg')
plt.show()
