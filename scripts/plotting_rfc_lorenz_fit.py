import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr


def kld(x: np.ndarray, y: np.ndarray, n_bins: int = 100) -> tuple:

    # get histograms of arrays
    x_hist, bin_edges = np.histogram(x, bins=n_bins)
    y_hist, _ = np.histogram(y, bins=bin_edges)

    # calculate KLD
    kl_div = np.sum(rel_entr(x, y))
    return kl_div, x_hist, y_hist, bin_edges


# load data
file = "alpha1_0"
data = pickle.load(open(f"../results/rfc_lorenz/{file}.pkl", "rb"))
n_in = data["targets"].shape[1]

# calculate difference between histograms
n_bins = 200
distributions = []
for i in range(n_in):
    targets = data["targets"][:, i]
    predictions = data["predictions"][:, i]
    kl_div, pred_dist, targ_dist, edges = kld(predictions, targets, n_bins=n_bins)
    distributions.append({"kld": kl_div, "predictions": pred_dist, "targets": targ_dist, "edges": edges})

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
    ax.set_ylabel("count")

plt.tight_layout()
plt.show()
