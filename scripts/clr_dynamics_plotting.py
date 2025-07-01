import numpy as np
import pickle
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
data_set = "dendritic_gain"
path = "/home/richard-gast/Documents"
results = read_csv(f"{path}/results/{data_set}_dynamics.csv")

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# line plots for a single in-scale
##################################

# choose a particular parameter
choose_param = "Delta"
value = 0.6
results_tmp = results.loc[results.loc[:, choose_param] == value, :]

# reduce data to the plotting selection
hue_param = "lambda"
hue_vals = np.unique(results_tmp.loc[:, hue_param].values)
sigmas = np.unique(results_tmp.loc[:, "sigma"].values)

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
sb.lineplot(results_tmp, ax=ax, x="sigma", y="lyapunov", hue=hue_param, err_style="band", palette="viridis")
zero_crossings = []
for line, _ in zip(ax.get_lines(), hue_vals):
    x, y = line.get_data()
    zero_crossings.append(x[np.argmin(np.abs(y))])
ax.set_ylim([-0.06, 0.025])
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$\lambda$")
ax.set_title(r"Maximum Lyapunov Exponent $\lambda$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/{data_set}_le.svg")

# create MC figure
fig, ax = plt.subplots()
# sb.heatmap(C, ax=ax, cmap="cividis")
# ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
# ax.set_ylabel(r"neural heterogeneity $\Delta$")
# ax.set_title("Memory Capacity")
sb.lineplot(results_tmp, ax=ax, x="sigma", y="memory", hue=hue_param, err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$C$")
ax.set_title(r"Memory Capacity $C$")
ax.set_yticklabels([np.round(float(l._y), decimals=1) for l in ax.get_yticklabels()])
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/{data_set}_mc.svg")

# create TS figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="entropy", hue=hue_param, err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$H$")
ax.set_title(r"Timescale Heterogeneity $H$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/{data_set}_ts.svg")

# create PSD figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="psd", hue=hue_param, err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$\rho$")
ax.set_title(r"Spectral Participation Ratio $D_f$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/{data_set}_spectral_pr.svg")

# create PR figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="dimensionality", hue=hue_param, err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$D$")
ax.set_title(r"Participation Ratio $D$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/{data_set}_pr.svg")

plt.show()
