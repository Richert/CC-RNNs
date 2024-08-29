import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pickle
import sys
import os

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

        # get condition
        results["alpha"].append(data["alpha"])
        results["motifs"].append(data["motifs"])
        results["motif_length"].append(data["motif_length"])

        # calculate error between predictions and targets for input-driven mode
        predictions = data["input_predictions"]
        targets = data["input_targets"]
        results["input_error"].append(np.mean((predictions - targets)**2))

        # calculate error between predictions and targets for autonomous mode
        predictions = data["sequence_predictions"]
        targets = data["sequence_targets"]
        results["sequence_error"].append(np.mean((predictions - targets) ** 2))

df = DataFrame.from_dict(results)
n_motifs = len(np.unique(df["motifs"].values))
n_motif_lengths = len(np.unique(df["motif_length"].values))

# plot results
fig = plt.figure(figsize=(12, 9))
grid = fig.add_gridspec(nrows=n_motifs, ncols=n_motif_lengths)
