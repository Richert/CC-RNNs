import numpy as np
import pickle
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
path = "/home/richard-gast/Documents"
data = pickle.load(open(f"{path}/data/clr_dynamics.pkl", "rb"))
results = read_csv(f"{path}/results/clr_dynamics.csv")

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# plot raw dynamics
###################

n_examples = 1
Delta = [0.0]
sigma = [1.9]
for Delta_tmp, sigma_tmp in zip(Delta, sigma):

    # get index of condition
    delta_idx = np.asarray(data["Delta"]) == Delta_tmp
    sigma_idx = np.asarray(data["sigma"]) == sigma_tmp
    final_idx = 1.0 * delta_idx * sigma_idx

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

# line plots
############

# choose input strength
in_scale = 0.1
results = results.loc[results.loc[:, "in_scale"] == in_scale, :]

# reduce data to the plotting selection
deltas = np.unique(results.loc[:, "Delta"].values)
sigmas = np.unique(results.loc[:, "sigma"].values)

# create 2D matrix of MC, PR, and TS
# C = np.zeros((len(deltas), len(sigmas)))
# TS = np.zeros_like(C)
# PR = np.zeros_like(C)
# for i, Delta in enumerate(deltas):
#     for j, sigma in enumerate(sigmas):
#         idx1 = results.loc[:, "Delta"].values == Delta
#         idx2 = results.loc[:, "sigma"].values == sigma
#         C[i, j] = np.mean(results.loc[idx1 & idx2, "memory"].values)
#         TS[i, j] = np.mean(results.loc[idx1 & idx2, "timescale_heterogeneity"].values)
#         PR[i, j] = np.mean(results.loc[idx1 & idx2, "dimensionality"].values)
# cols = np.round(sigmas, decimals=1)
# indices = np.round(deltas, decimals=2)
# C = DataFrame(columns=cols, index=indices, data=C)
# TS = DataFrame(columns=cols, index=indices, data=TS)
# PR = DataFrame(columns=cols, index=indices, data=PR)

# create LE figure
fig, ax = plt.subplots()
sb.lineplot(results, ax=ax, x="sigma", y="lyapunov", hue="Delta", err_style="band", palette="viridis")
zero_crossings = []
for line, _ in zip(ax.get_lines(), deltas):
    x, y = line.get_data()
    zero_crossings.append(x[np.argmin(np.abs(y))])
ax.set_ylim([-0.06, 0.025])
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$\lambda$")
ax.set_title(r"Maximum Lyapunov Exponent $\lambda$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_le.svg")

# create MC figure
fig, ax = plt.subplots()
# sb.heatmap(C, ax=ax, cmap="cividis")
# ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
# ax.set_ylabel(r"neural heterogeneity $\Delta$")
# ax.set_title("Memory Capacity")
sb.lineplot(results, ax=ax, x="sigma", y="memory", hue="Delta", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$C$")
ax.set_title(r"Memory Capacity $C$")
ax.set_yticklabels([np.round(float(l._y), decimals=1) for l in ax.get_yticklabels()])
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_mc.svg")

# create TS figure
fig, ax = plt.subplots()
sb.lineplot(results, ax=ax, x="sigma", y="timescale_heterogeneity", hue="Delta", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$H$")
ax.set_title(r"Timescale Heterogeneity $\Delta$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_ts.svg")

# create PR figure
fig, ax = plt.subplots()
sb.lineplot(results, ax=ax, x="sigma", y="dimensionality", hue="Delta", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$D$")
ax.set_title(r"Participation Ratio $D$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_pr.svg")

plt.show()
