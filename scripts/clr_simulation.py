import torch
import numpy as np
from src.rnn import LowRankCRNN
from src.functions import init_weights, init_dendrites
import matplotlib.pyplot as plt

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"

# simulation parameters
steps = 1000

# rnn parameters
n_in = 1
k = 200
n_dendrites = 10
N = int(k*n_dendrites)
in_scale = 0.1
density = 0.5
sigma1 = 0.0
sigma2 = 2.0
sigmas = np.linspace(sigma1, sigma2, num=steps)
Deltas = [0.0, 0.1, 0.2, 0.4]

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(np.random.randn(k), device=device, dtype=dtype)
L = init_weights(N, k, density)
W, R = init_dendrites(k, n_dendrites)

# input definition
inp = torch.randn((steps, n_in), device=device, dtype=dtype)
inp[int(0.5*steps):, :] = 0.0

# simulation
############

results = []
for Delta in Deltas:

    # model initialization
    rnn = LowRankCRNN(torch.tensor(W, dtype=dtype, device=device), torch.tensor(L, dtype=dtype, device=device),
                      torch.tensor(R, device=device, dtype=dtype), W_in, bias*Delta, g="ReLU")

    # model dynamics simulation
    z_col = []
    with torch.no_grad():
        for step in range(steps):
            rnn.C_z = sigmas[step]
            rnn.forward(inp[step])
            z_col.append(rnn.y.detach().cpu().numpy())
    results.append(np.asarray(z_col))

# plotting
##########

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12.0, 2.0*len(Deltas))
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

fig, axes = plt.subplots(nrows=len(Deltas))
neurons = np.random.choice(N, size=(100,))
for idx, Delta in enumerate(Deltas):
    ax = axes[idx]
    ax.imshow(results[idx][:, neurons].T, aspect="auto", interpolation="none", cmap="Greys")
    ax.set_ylabel("neuron")
    if idx == 1:
        ax.set_xlabel("time")
    ax.set_title(rf"Network Dynamics for $\Delta = {np.round(Delta, decimals=1)}$")
plt.tight_layout(pad=0.5)
fig.canvas.draw()
fig.savefig("/home/richard-gast/Documents/results/clr_dynamics.svg")
plt.show()
