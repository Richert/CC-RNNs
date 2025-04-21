import numpy as np
import pickle
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
path = "/home/richard-gast/Documents"
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

# line plots for a single in-scale
##################################

# choose input strength
in_scale = 0.01
results_tmp = results.loc[results.loc[:, "in_scale"] == in_scale, :]

# reduce data to the plotting selection
deltas = np.unique(results_tmp.loc[:, "Delta"].values)
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
sb.lineplot(results_tmp, ax=ax, x="sigma", y="lyapunov", hue="Delta", err_style="band", palette="viridis")
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
sb.lineplot(results_tmp, ax=ax, x="sigma", y="memory", hue="Delta", err_style="band", palette="viridis")
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
sb.lineplot(results_tmp, ax=ax, x="sigma", y="entropy", hue="Delta", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$H$")
ax.set_title(r"Timescale Heterogeneity $H$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_ts.svg")

# create PSD figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="psd", hue="Delta", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$\rho$")
ax.set_title(r"Power Spectral Density $\rho$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_psd.svg")

# create PR figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="dimensionality", hue="Delta", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$D$")
ax.set_title(r"Participation Ratio $D$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_pr.svg")

# line plots for a single Delta
###############################

# choose input strength
Delta = 0.0
results_tmp = results.loc[results.loc[:, "Delta"] == Delta, :]

# reduce data to the plotting selection
in_scales = np.unique(results.loc[:, "in_scale"].values)
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
sb.lineplot(results_tmp, ax=ax, x="sigma", y="lyapunov", hue="in_scale", err_style="band", palette="viridis")
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
fig.savefig(f"{path}/results/clr_le2.svg")

# create MC figure
fig, ax = plt.subplots()
# sb.heatmap(C, ax=ax, cmap="cividis")
# ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
# ax.set_ylabel(r"neural heterogeneity $\Delta$")
# ax.set_title("Memory Capacity")
sb.lineplot(results_tmp, ax=ax, x="sigma", y="memory", hue="in_scale", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$C$")
ax.set_title(r"Memory Capacity $C$")
ax.set_yticklabels([np.round(float(l._y), decimals=1) for l in ax.get_yticklabels()])
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_mc2.svg")

# create TS figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="entropy", hue="in_scale", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$H$")
ax.set_title(r"Timescale Heterogeneity $H$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_ts2.svg")

# create PSD figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="psd", hue="in_scale", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$\rho$")
ax.set_title(r"Power Spectral Density $\rho$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_psd2.svg")

# create PR figure
fig, ax = plt.subplots()
sb.lineplot(results_tmp, ax=ax, x="sigma", y="dimensionality", hue="in_scale", err_style="band", palette="viridis")
for line, idx in zip(ax.get_lines(), zero_crossings):
    ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
ax.set_ylabel(r"$D$")
ax.set_title(r"Participation Ratio $D$")
plt.tight_layout()
fig.canvas.draw()
fig.savefig(f"{path}/results/clr_pr2.svg")

plt.show()
