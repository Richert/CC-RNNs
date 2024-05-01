import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from scipy.stats import wasserstein_distance
import os
import pandas as pd


def wasserstein(x: np.ndarray, y: np.ndarray, n_bins: int = 100) -> tuple:

    # get histograms of arrays
    x_hist, x_edges = np.histogram(x, bins=n_bins, density=True)
    y_hist, y_edges = np.histogram(y, bins=n_bins, density=True)
    x_hist /= np.sum(x_hist)
    y_hist /= np.sum(y_hist)

    # calculate KLD
    wd = wasserstein_distance(u_values=x_edges[:-1], v_values=y_edges[:-1], u_weights=x_hist, v_weights=y_hist)
    return wd, x_hist, y_hist, x_edges, y_edges


# data collection and analysis
##############################

files = [f for f in os.listdir("../results/lr_lorenz") if f[0] == "k"]
models = ["rfc", "lr", "clr"]
df = pd.DataFrame(columns=["model", "steps", "rep", "wd", "k_star", "dim", "train_error"],
                  index=np.arange(0, len(files)), dtype=np.float64)
n_bins = 1000
eps = 1e-10
n = 0
for file in files:
    for model in models:

        # load data
        data = pickle.load(open(f"../results/{model}_lorenz/{file}", "rb"))
        steps = data["condition"]["steps"]
        rep = data["condition"]["repetition"]
        error = data["training_error"]

        # calculate dimensionality
        k_star = np.sum(data["c"])
        dim = np.sum(data["c"] > eps)

        # calculate wasserstein distance
        wd = 0.0
        for i in range(data["targets"].shape[1]):
            targets = data["targets"][:, i]
            predictions = data["predictions"][:, i]
            wd_tmp, *_ = wasserstein(predictions, targets, n_bins=n_bins)
            wd += wd_tmp

        df.loc[n, :] = (steps, rep, wd, k_star, dim, error)
        n += 1

# collect representative trajectories for target steps
steps = [100000, 300000, 500000]
trajectories = {model: [] for model in models}
for n_steps in steps:

    df_tmp = df.where(df.loc[:, "steps"] == n_steps)

    for model in models:

        # find representative sample
        df_tmp = df_tmp.where(df.loc[:, "model"] == model)
        error = np.mean(df_tmp.loc[:, "train_error"].values)
        idx = np.argmin(np.abs(df_tmp.loc[:, "train_error"].values - error))
        rep = df_tmp.at[idx, "rep"]

        # load data
        data = pickle.load(open(f"../results/{model}_lorenz/n{n_steps}_{int(rep)}.pkl", "rb"))
        trajectories[model].append(data["predictions"])

# get target trajectory
targets = pickle.load(open(f"../results/lr_lorenz/{files[0]}", "rb"))["targets"]

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
subfigs = fig.subfigures(nrows=2, height_ratios=[0.3, 0.7])

# violin plots
x = "steps"
ys = ["wd", "train_error"]
titles = ["Wasserstein Distance (test data)", "MSE (training data)"]
axes = subfigs[0].subplots(ncols=len(ys))
for i in range(len(ys)):

    ax = axes[i]
    sb.lineplot(x=x, y=ys[i], hue="model", data=df, color="0.8", ax=ax)
    sb.stripplot(x=x, y=ys[i], hue="model", data=df, jitter=True, zorder=1, ax=ax)
    ax.set_xlabel("steps")
    ax.set_ylabel(titles[i])

subfigs[0].suptitle("Model performance")

# trajectory plots
titles = [f"steps = {n}" for n in steps] + ["Lorenz equations"]
colors = ["darkgreen", "darkblue", "darkred"]
model_titles = ["RFC", "LR-RNN", "cLR- RNN"]
lvars = [0, 2]
axes = subfigs[1].subplots(ncols=len(titles), nrows=len(models))
for i in range(len(models)):
    for j in range(len(titles)):

        ax = axes[i, j]
        if j < len(titles) - 1:
            data = trajectories[models[i]][j]
            c = colors[i]
        else:
            data = targets
            c = "black"

        ax.plot(data[plot_start:plot_stop, lvars[0]], data[plot_start:plot_stop, lvars[1]], color=c)
        ax.set_xlabel(fr"$u_{lvars[0] + 1}$")
        ax.set_ylabel(fr"$u_{lvars[1] + 1}$")
        ax.set_title(f"{model_titles[i]}: {titles[j]}")

subfigs[1].suptitle("Lorenz attractor reconstructions")

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.02)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'../results/lorenz_all.svg')
plt.show()
