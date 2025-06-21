import sys
sys.path.append('../')
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

# task parameters
steps = 500
washout = 100
epsilon = 1e-5

# rnn parameters
k = 200
n_in = 1
n_dendrites = 10
density = 0.5
in_scale = 0.2
N = int(k * n_dendrites)
sigma = 1.0
lam = 3.2
Delta = 0.0

# simulations
#############

with torch.no_grad():

    # initialize rnn matrices
    bias = torch.zeros(k, device=device, dtype=dtype)
    W_in = torch.tensor(np.random.randn(N, n_in), device=device, dtype=dtype)
    L = torch.tensor(init_weights(N, k, density), device=device, dtype=dtype)
    W, R = init_dendrites(k, n_dendrites)
    R = torch.tensor(R, device=device, dtype=dtype)

    # input definition
    inp = torch.randn((steps, n_in), device=device, dtype=dtype)

    # get initial and perturbed state
    successful = False
    while not successful:
        init_state = [2*torch.rand(N, device=device, dtype=dtype) - 1.0,
                      2.0*torch.rand(k, device=device, dtype=dtype)]
        perturbed_state = [v[:] + epsilon * torch.randn(v.shape[0]) for v in init_state]
        diffs = [torch.sum((v - v_p) ** 2) for v, v_p in zip(init_state, perturbed_state)]
        if all([d.item() > 0 for d in diffs]):
            successful = True

    # model initialization
    W_tmp = torch.tensor(W*lam, dtype=dtype, device=device)
    dendritic_gains = np.random.uniform(low=1.0-Delta, high=1.0+Delta, size=N)
    rnn = LowRankCRNN(W_tmp, L*sigma, R, W_in * in_scale, bias, g="ReLU")
    rnn.C_y = torch.tensor(dendritic_gains, dtype=dtype, device=device)

    # simulation a - zero input
    rnn.set_state(init_state)
    z0s = []
    x = torch.zeros(n_in, dtype=dtype, device=device)
    for step in range(steps):
        z0s.append(rnn.z.detach().cpu().numpy())
        rnn.forward(x)

    # simulation b - random input
    rnn.set_state(init_state)
    z1s = []
    for step in range(steps):
        z1s.append(rnn.z.detach().cpu().numpy())
        rnn.forward(inp[step])

    # model simulation II
    rnn.set_state(perturbed_state)
    z2s = []
    for step in range(steps):
        z2s.append(rnn.z.detach().cpu().numpy())
        rnn.forward(inp[step])

# plotting
##########

cmap = "plasma"
fig = plt.figure(figsize=(12, 9))
grid = fig.add_gridspec(nrows=3)

ax = fig.add_subplot(grid[0])
im = ax.imshow(np.asarray(z0s)[washout:].T, aspect="auto", interpolation="none", cmap=cmap)
plt.colorbar(im, ax=ax)
ax.set_ylabel("neurons")
ax.set_title("no input")

ax = fig.add_subplot(grid[1])
im = ax.imshow(np.asarray(z1s)[washout:].T, aspect="auto", interpolation="none", cmap=cmap)
plt.colorbar(im, ax=ax)
ax.set_ylabel("neurons")
ax.set_title("random input")

ax = fig.add_subplot(grid[2])
im = ax.imshow(np.asarray(z2s)[washout:].T, aspect="auto", interpolation="none", cmap=cmap)
plt.colorbar(im, ax=ax)
ax.set_ylabel("neurons")
ax.set_title("random input + initial state perturbation")
ax.set_xlabel("steps")
plt.tight_layout()
plt.show()
