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


# get effective dimensionality and reconstruction quality for each condition
files = os.listdir("../results/rfc_lorenz")
df = pd.DataFrame(columns=["alpha", "rep", "wd", "k", "dim"], index=np.arange(0, len(files)))
n_bins = 200
for n, file in enumerate(files):

    # load data
    data = pickle.load(open(f"../results/rfc_lorenz/{file}", "rb"))
    alpha = data["condition"]["alpha"]
    rep = data["condition"]["repitition"]

    # calculate dimensionality
    k_star = np.sum(data["c"])
    dim = np.sum(data["c"] > 0.0)

    # calculate wasserstein distance
    wd = 0.0
    for i in range(data["targets"].shape[1]):
        targets = data["targets"][:, i]
        predictions = data["predictions"][:, i]
        wd_tmp, *_ = wasserstein(targets, predictions, n_bins=n_bins)
        wd += wd_tmp

    df.loc[n, :] = (alpha, rep, wd, k_star, dim)
