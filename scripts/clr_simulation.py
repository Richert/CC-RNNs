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
steps = 200

# rnn parameters
n_in = 1
n_out = 1
k = 100
n_dendrites = 10
N = int(k*n_dendrites)
in_scale = 0.2
density = 0.5
g_w = 0.0
sigma = 0.4
Delta = 0.05

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
rnn.C_z *= sigma

# input definition
inp = torch.zeros((steps, n_in), device=device, dtype=dtype)
# inp[0, :] = 1.0

# model dynamics simulation
y_col, z_col = [], []
with torch.no_grad():
    for step in range(steps):
        rnn.forward(inp[step])
        y_col.append(rnn.y.detach().cpu().numpy())
        z_col.append(rnn.z.detach().cpu().numpy())
y_col = np.asarray(y_col)
z_col = np.asarray(z_col)

# plotting
##########

# dynamics
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
for i in range(N):
    ax.plot(y_col[:, i], label=f"dendrite {i}")
ax.set_xlabel("time")
ax.set_ylabel("y")
ax.set_title("dendritic dynamics")
ax = axes[1]
for i in range(k):
    ax.plot(z_col[:, i], label=f"soma {i}")
ax.set_xlabel("time")
ax.set_ylabel("z")
ax.legend()
ax.set_title("somatic dynamics")
plt.tight_layout()
plt.show()
