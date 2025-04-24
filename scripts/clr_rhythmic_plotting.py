import numpy as np
import pickle
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
path = "/home/richard-gast/Documents"
task = "clr_rhythmic_3freq"
conditions = ["cfit"]

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# collect data
results = {"condition": [], "sigma": [], "Delta": [], "trial": [], "train_epochs": [], "train_loss": [], "test_loss": []}
for cond in conditions:
    data = pickle.load(open(f"{path}/{task}_{cond}.pkl", "rb"))
    for key, val in data.items():
        results[key].extend(val)
    results[cond].extend([cond]*len(data["trial"]))
df = DataFrame.from_dict(results)

# find
deltas = np.unique(df.loc[:, "Delta"].values)

# create figure
fig = plt.figure()
grid = fig.add_gridspec(nrows=len(deltas), ncols=2)
for i, Delta in enumerate(deltas):

    # constrain data to particular Delta
    df_tmp = df.loc[df.loc[:, "Delta"] == Delta, :]

    # plot number of training epochs
    ax = fig.add_subplot(grid[i, 0])
    sb.lineplot(df_tmp, x="sigma", y="train_epochs", hue="condition")
    ax.title("Number of training epochs")
    ax.set_ylabel("epochs")
    ax.set_xlabel(r"$\sigma$")

    # plot test loss
    ax = fig.add_subplot(grid[i, 1])
    sb.lineplot(df_tmp, x="sigma", y="test_loss", hue="condition")
    ax.title("Loss on test trials")
    ax.set_ylabel("MSE")
    ax.set_xlabel(r"$\sigma$")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'/home/richard-gast/Documents/results/clr_fitting.svg')
plt.show()
