import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pickle
import sys
import os
import seaborn as sb

# condition
path = sys.argv[-3]
keys = sys.argv[-2]
fingers = sys.argv[-1]
identifier = f"piano_crnn_{keys}keys_{fingers}fingers"

# process simulation data into results
results = {"input_error": [], "sequence_error": [], "alpha": [], "motifs": [], "motif_length": []}

for f in os.listdir(path):
    if identifier in f:

        # load data
        data = pickle.load(open(f"{path}/{f}", "rb"))

        try:

            # get condition
            results["alpha"].append(data["alpha"])
            results["motifs"].append(data["motifs"])
            results["motif_length"].append(data["motif_length"])

            # calculate error between predictions and targets for input-driven mode
            predictions = np.asarray(data["input_predictions"]).flatten()
            targets = np.asarray(data["input_targets"]).flatten()
            results["input_error"].append(np.mean((predictions - targets)**2))

            # calculate error between predictions and targets for autonomous mode
            predictions = np.asarray(data["sequence_predictions"]).flatten()
            targets = np.asarray(data["sequence_targets"]).flatten()
            results["sequence_error"].append(np.mean((predictions - targets) ** 2))

        except KeyError:

            print(list(data.keys()))

df = DataFrame.from_dict(results)
motifs = np.unique(df["motifs"].values)
motif_lengths = np.unique(df["motif_length"].values)
n_motifs = len(motifs)
n_motif_lengths = len(motif_lengths)

# plot input-driven results
fig = plt.figure(figsize=(12, 3))
grid = fig.add_gridspec(nrows=1, ncols=n_motif_lengths)
for i in range(n_motif_lengths):
    l = motif_lengths[i]
    df_tmp = df.loc[df["motif_length"] == l, :]
    ax = fig.add_subplot(grid[0, i])
    sb.lineplot(data=df_tmp, x="alpha", y="input_error", hue="motifs")
    ax.set_xlabel("alpha")
    ax.set_xscale("log")
    if i == 0:
        ax.set_ylabel("MSE")
fig.suptitle("Input-driven motor command mode")
plt.tight_layout()

# plot sequence generation results
fig = plt.figure(figsize=(12, 3))
grid = fig.add_gridspec(nrows=1, ncols=n_motif_lengths)
for i in range(n_motif_lengths):
    l = motif_lengths[i]
    df_tmp = df.loc[df["motif_length"] == l, :]
    ax = fig.add_subplot(grid[0, i])
    sb.lineplot(data=df_tmp, x="alpha", y="sequence_error", hue="motifs")
    ax.set_xlabel("alpha")
    ax.set_xscale("log")
    ax.set_title(f"Motif length: {i+1}")
    if i == 0:
        ax.set_ylabel("MSE")
fig.suptitle("Autonomous motor sequence generation mode")
plt.tight_layout()

plt.show()
