import torch
import numpy as np
from src.rnn import FHNRNN
from src.functions import init_weights, init_dendrites
import matplotlib.pyplot as plt

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"

# simulation parameters
dt = 1e-1
steps = 20000

# FHN parameters
a = 0.7
b = 0.8
tau = 12.5
gamma = 100.0
theta = 1.6

# rnn parameters
n_in = 1
n_out = 1
k = 5
n_dendrites = 5
N = int(k*n_dendrites)
sr = 0.99
bias_scale = 0.01
in_scale = 0.02
out_scale = 0.02
density = 0.5

# initialize rnn matrices
W_in = torch.tensor(np.ones((N, n_in)), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
L = init_weights(N, k, density)
R = init_dendrites(k, n_dendrites)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(L, R))))
L *= np.sqrt(sr) / np.sqrt(sr_comb)
R *= np.sqrt(sr) / np.sqrt(sr_comb)

# simulation
############

# model initialization
rnn = FHNRNN(torch.tensor(L, dtype=dtype, device=device), torch.tensor(R, device=device, dtype=dtype), W_in, bias,
             a=a, b=b, tau=tau, gamma=gamma, theta=theta, dt=dt)

# input definition
inp = torch.zeros((steps, n_in), device=device, dtype=dtype) + 0.5
inp[int(0.3*steps):int(0.6*steps), :] += 0.1

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

fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
for i in range(N):
    ax.plot(y_col[:, i], label=f"dendrite {i}")
ax.set_xlabel("time")
ax.set_ylabel("v")
ax.set_title("dendritic dynamics")
ax = axes[1]
for i in range(k):
    ax.plot(z_col[:, i], label=f"soma {i}")
ax.set_xlabel("time")
ax.set_ylabel("v")
ax.legend()
ax.set_title("somatic dynamics")
plt.tight_layout()
plt.show()
