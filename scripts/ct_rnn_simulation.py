from src.ct_rnn import LowRankRNN
from src.functions import init_weights
import torch
import matplotlib.pyplot as plt
import numpy as np

# general
dtype = torch.float64
device = "cpu"

# network parameters
N = 200
n_in = 1
k = 2
sr = 1.1
bias_min = -0.01
bias_max = 0.01
in_scale = 0.1
tau_min = 0.1
tau_max = 0.5
density = 0.2
out_scale = 0.2
init_noise = 0.2
dt = 0.001

# rnn matrices
W_in = torch.tensor(in_scale * np.random.rand(N, n_in), device=device, dtype=dtype)
W = torch.tensor(sr * init_weights(N, N, density))
bias = torch.tensor(bias_min + bias_max * np.random.rand(N), device=device, dtype=dtype)
taus = torch.tensor(tau_min + tau_max * np.random.rand(N), device=device, dtype=dtype)
L = init_weights(N, k, density)
R = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(L, R))))
L *= np.sqrt(0.0*sr) / np.sqrt(sr_comb)
R *= np.sqrt(0.0*sr) / np.sqrt(sr_comb)

# initialize LR-RNN
rnn = LowRankRNN(W, W_in, bias, taus, torch.tensor(L, device=device, dtype=dtype),
                 torch.tensor(R, device=device, dtype=dtype), dt=dt)

# define input
steps = 1000
amp = 2.0
freq = 10.0
time = np.arange(0, steps) * dt
sine = np.sin(freq*2.0*np.pi*time)
inp = np.zeros((steps, n_in))
inp[:, 0] = amp * sine
inp = torch.tensor(inp, dtype=dtype, device=device)

# perform simulation
results = []
with torch.no_grad():
    for step in range(steps):
        results.append(rnn.forward(inp[step]).cpu().detach().numpy())
results = np.asarray(results)

# plotting
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=2, height_ratios=[1.0, 0.5])
ax = fig.add_subplot(grid[0])
im = ax.imshow(results.T, interpolation="none", aspect="auto", cmap="viridis")
plt.colorbar(im, ax=ax, shrink=0.65)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
ax = fig.add_subplot(grid[1])
ax.plot(np.mean(results, axis=1))
ax.set_xlabel("steps")
ax.set_ylabel("mean(y)")
plt.tight_layout()
plt.show()
