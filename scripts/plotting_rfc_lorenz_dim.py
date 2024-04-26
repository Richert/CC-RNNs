import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from scipy.stats import wasserstein_distance
import os


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
results = {}
for file in os.listdir("../results/rfc_lorenz"):

    data = pickle.load(open(f"../results/rfc_lorenz/{file}.pkl", "rb"))
    alpha = data["condition"]["alpha"]
    if alpha in results:
        results[alpha].append(data)
    else:
        results[alpha] = [data]

# get effective dimensionality and reconstruction quality as a function of alpha
k_stars = []
dimensionalities = []
distances = []
alphas = list(results.keys())
n_bins = 200
for alpha in alphas:

    k_col = []
    dim_col = []
    wd_col = []

    for data in results[alpha]:

        k_col.append(np.sum(data["c"]))
        dim_col.append(np.sum(data["c"] > 0.0))

        wd = 0.0
        for i in range(data["targets"].shape[1]):
            targets = data["targets"][:, i]
            predictions = data["predictions"][:, i]
            wd_tmp, *_ = wasserstein(targets, predictions, n_bins=n_bins)
            wd += wd_tmp
        wd_col.append(wd)

    k_stars.append(k_col)
    dimensionalities.append(dim_col)
    distances.append(wd_col)
