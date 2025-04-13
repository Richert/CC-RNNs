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
steps = 500

# rnn parameters
n_in = 1
n_out = 1
k = 100
n_dendrites = 10
N = int(k*n_dendrites)
in_scale = 0.2
density = 0.5
g_w = 0.0
sigma1 = 0.5
sigma2 = 3.0
sigmas = np.linspace(sigma1, sigma2, num=steps)
Delta = 0.3

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(Delta * np.random.randn(k), device=device, dtype=dtype)
L = init_weights(N, k, density)
W, R = init_dendrites(k, n_dendrites)

# simulation
############

# model initialization
rnn = LowRankCRNN(torch.tensor(W*g_w, dtype=dtype, device=device), torch.tensor(L, dtype=dtype, device=device),
                  torch.tensor(R, device=device, dtype=dtype), W_in, bias, g="ReLU")

# input definition
inp = torch.zeros((steps, n_in), device=device, dtype=dtype)

# model dynamics simulation
y_col, z_col = [], []
with torch.no_grad():
    for step in range(steps):
        rnn.C_z = sigmas[step]
        rnn.forward(inp[step])
        y_col.append(rnn.y.detach().cpu().numpy())
        z_col.append(rnn.z.detach().cpu().numpy())
y_col = np.asarray(y_col)
z_col = np.asarray(z_col)

# plotting
##########

# dynamics
fig, ax = plt.subplots(figsize=(10, 2))
ax.imshow(z_col.T, aspect="auto", interpolation="none", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neuron")
plt.tight_layout()

plt.show()
