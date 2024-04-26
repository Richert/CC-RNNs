import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


def wasserstein(x: np.ndarray, y: np.ndarray, n_bins: int = 100) -> tuple:

    # get histograms of arrays
    x_hist, bin_edges = np.histogram(x, bins=n_bins, density=True)
    y_hist, _ = np.histogram(y, bins=bin_edges, density=True)
    x_hist /= np.sum(x_hist)
    y_hist /= np.sum(y_hist)

    # calculate KLD
    wd = wasserstein_distance(x_hist, y_hist)
    return wd, x_hist, y_hist, bin_edges


# load data
file = "alpha1_1"
data = pickle.load(open(f"../results/rfc_lorenz/{file}.pkl", "rb"))
n_in = data["targets"].shape[1]

# calculate difference between histograms
n_bins = 200
distributions = []
for i in range(n_in):
    targets = data["targets"][:10000, i]
    predictions = data["predictions"][:, i]
    wd, pred_dist, targ_dist, edges = wasserstein(predictions, targets, n_bins=n_bins)
    distributions.append({"wd": wd, "predictions": pred_dist, "targets": targ_dist, "edges": edges})

# plotting of trajectories in state space
plot_steps = 4000
fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))
for i, ax in enumerate(axes):

    ax.plot(data["targets"][:plot_steps, i], color="royalblue", label="target")
    ax.plot(data["predictions"][:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(f"u_{i+1}")
    if i == n_in-1:
        ax.set_xlabel("steps")
        ax.legend()

plt.tight_layout()

# plotting of distributions
fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))
for i, ax in enumerate(axes):

    ax.bar(distributions[i]["edges"][:-1], distributions[i]["targets"], width=0.5, align="edge",
           color="royalblue", label="target")
    ax.bar(distributions[i]["edges"][1:], distributions[i]["predictions"], width=-0.5, align="edge",
           color="darkorange", label="prediction")
    ax.set_xlabel(f"u_{i+1}")
    ax.set_ylabel("p")
    ax.set_title(f"Wasserstein distance = {distributions[i]['wd']}")
    if i == n_in - 1:
        ax.legend()

plt.tight_layout()
plt.show()
