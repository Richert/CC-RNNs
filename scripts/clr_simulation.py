import torch
import numpy as np
from src.rnn import LowRankCRNN
from src.functions import init_weights, init_dendrites
import matplotlib.pyplot as plt
import seaborn as sb
from pandas import read_csv

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"

# simulation parameters
sigmas = [0.1, 0.5, 1.5]
period = 100
steps = int(len(sigmas)*period)
init_steps = 20

# rnn parameters
n_in = 1
k = 200
n_dendrites = 10
N = int(k*n_dendrites)
in_scale = 0.6
density = 0.5
Delta = 0.1

# initialize rnn matrices
W_in = torch.tensor(np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(np.random.randn(k), device=device, dtype=dtype)
L = init_weights(N, k, density)
W, R = init_dendrites(k, n_dendrites)

# simulation
############

# model initialization
rnn = LowRankCRNN(torch.tensor(W, dtype=dtype, device=device), torch.tensor(L, dtype=dtype, device=device),
                  torch.tensor(R, device=device, dtype=dtype), in_scale*W_in, bias*Delta, g="ReLU")

# model dynamics simulation
z_col = []
with torch.no_grad():
    for sigma in sigmas:
        rnn.C_z = sigma
        inp = torch.zeros((period, n_in), device=device, dtype=dtype)
        inp[int(0.5*period), :] = 1.0
        for step in range(init_steps):
            rnn.forward(inp[0])
        for step in range(period):
            rnn.forward(inp[step])
            z_col.append(rnn.z.detach().cpu().numpy())
results = np.asarray(z_col)
for i in range(len(sigmas)):
    for j in range(results.shape[1]):
        max_fr = np.max(results[i*period:(i+1)*period, j])
        if max_fr > 0.0:
            results[i*period:(i+1)*period, j] /= max_fr

# plotting
##########

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12.0, 7.0)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure layout
fig = plt.figure()
grid = fig.add_gridspec(nrows=3, ncols=3)

# load data
path = "/home/richard-gast/Documents"
data = read_csv(f"{path}/results/clr_dynamics.csv")

for n, (val, param, other_param) in enumerate(zip([0.1, 0.1], ["in_scale", "Delta"], ["Delta", "in_scale"])):

    results_tmp = data.loc[data.loc[:, param] == val, :]

    # reduce data to the plotting selection
    # sigmas = np.unique(results_tmp.loc[:, "sigma"].values)
    deltas = np.unique(results_tmp.loc[:, other_param].values)

    # # create 2D matrix of MC, PR, and TS
    # MC = np.zeros((len(deltas), len(sigmas)))
    # PR = np.zeros_like(MC)
    # for i, Delta in enumerate(deltas):
    #     for j, sigma in enumerate(sigmas):
    #         idx1 = results_tmp.loc[:, other_param].values == Delta
    #         idx2 = results_tmp.loc[:, "sigma"].values == sigma
    #         MC[i, j] = np.mean(results_tmp.loc[idx1 & idx2, "memory"].values)
    #         PR[i, j] = np.mean(results_tmp.loc[idx1 & idx2, "dimensionality"].values)
    # cols = np.round(sigmas, decimals=1)
    # indices = np.round(deltas, decimals=2)

    # LE lineplot
    ax = fig.add_subplot(grid[n, 0])
    sb.lineplot(results_tmp, ax=ax, x="sigma", y="lyapunov", hue=other_param, err_style="band", palette="viridis")
    zero_crossings = []
    for line, _ in zip(ax.get_lines(), deltas):
        x, y = line.get_data()
        zero_crossings.append(x[np.argmin(np.abs(y))])
    ax.set_ylim([-0.06, 0.025])
    ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Maximum Lyapunov Exponent $\lambda$")

    # MC lineplot
    ax = fig.add_subplot(grid[n, 1])
    sb.lineplot(results_tmp, ax=ax, x="sigma", y="memory", hue=other_param, err_style="band", palette="viridis")
    for line, idx in zip(ax.get_lines(), zero_crossings):
        ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
    ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
    ax.set_ylabel(r"$C$")
    ax.set_title(r"Memory Capacity $C$")
    ax.set_yticklabels([np.round(float(l._y), decimals=1) for l in ax.get_yticklabels()])

    # PR lineplot
    ax = fig.add_subplot(grid[n, 2])
    sb.lineplot(results_tmp, ax=ax, x="sigma", y="dimensionality", hue=other_param, err_style="band", palette="viridis")
    for line, idx in zip(ax.get_lines(), zero_crossings):
        ax.axvline(x=idx, color=line.get_color(), linestyle="dotted")
    ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
    ax.set_ylabel(r"$D$")
    ax.set_title(r"Participation Ratio $D$")

    # heatmaps for MC and PR
    # for j, (title, ylabel, mat) in enumerate(zip([r"Memory Capacity $C$", r"Participation Ratio $D$"], [r"$C$", r"$D$"], [MC, PR])):
    #     ax = fig.add_subplot(grid[n, j+1])
    #     im = ax.imshow(mat, aspect="auto", interpolation="none", cmap="cividis")
    #     plt.colorbar(im, ax=ax)
    #     xticks = np.asarray(np.linspace(0, len(cols)-1, num=4), dtype=np.int16)
    #     ax.set_xticks(ticks=xticks, labels=cols[xticks])
    #     yticks = np.asarray(np.linspace(0, len(indices)-1, num=3), dtype=np.int16)
    #     ax.set_yticks(ticks=yticks, labels=indices[yticks])
    #     ax.set_xlabel(r"synaptic heterogeneity $\sigma$")
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(title)

# dynamics figures
ax = fig.add_subplot(grid[2, :])
vars = np.asarray([np.var(results[:, i]) for i in range(results.shape[1])])
neurons = np.random.permutation(np.argsort(vars)[-100:])
ax.imshow(results[:, neurons].T, aspect="auto", interpolation="none", cmap="BuPu")
ax.set_ylabel("neurons")
ax.set_xlabel("steps")
ax.set_title(rf"Network Dynamics for $\Delta = {np.round(Delta, decimals=1)}$")
# ax = fig.add_subplot(grid[3, :])
# for i in range(len(sigmas)):
#     mean_z = np.mean(results[i*period:(i+1)*period, :])
#     ax.plot(np.mean(results[i*period:(i+1)*period, :], axis=1) - mean_z, label=fr"$\sigma = {sigmas[i]}$")
# ax.set_ylabel(r"$\mathrm{mean}(z)$")
# ax.set_xlabel("steps")
# ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'/home/richard-gast/Documents/results/clr_dynamics.svg')
plt.show()
