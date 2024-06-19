from src import RandomFeatureConceptorRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights
from scipy.stats import wasserstein_distance


# function definitions
######################

def lorenz(x: float, y: float, z: float, s: float = 10.0, r: float = 28.0, b: float = 2.667) -> np.ndarray:
    """
    Parameters
    ----------
    x, y, z: float
        State variables.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : np.ndarray
       Vectorfield of the Lorenz equations.
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.asarray([x_dot, y_dot, z_dot])


def wasserstein(x: np.ndarray, y: np.ndarray, n_bins: int = 100) -> tuple:

    # get histograms of arrays
    x_hist, x_edges = np.histogram(x, bins=n_bins, density=True)
    y_hist, y_edges = np.histogram(y, bins=n_bins, density=True)
    x_hist /= np.sum(x_hist)
    y_hist /= np.sum(y_hist)

    # calculate KLD
    wd = wasserstein_distance(u_values=x_edges[:-1], v_values=y_edges[:-1], u_weights=x_hist, v_weights=y_hist)
    return wd, x_hist, y_hist, x_edges, y_edges


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 4000
state_vars = ["x", "y", "z"]
lag = 1
n_bins = 500

# lorenz equation parameters
s = 10.0
r = 28.0
b = 8/3
dt = 0.01
input_idx = np.asarray([0, 1, 2])
noise_lvl = 0.02

# reservoir parameters
N = 200
n_in = len(input_idx)
n_out = len(state_vars)
k = 600
sr = 0.99
bias_scale = 0.01
in_scale = 0.1
density = 0.2

# matrix initialization
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)

# training parameters
steps = 500000
init_steps = 1000
test_steps = 10000
loading_steps = 100000
lam = 0.002
betas = (0.9, 0.999)
alphas = (15.0, 1e-3)

# generate inputs and targets
#############################

# simulation
y = np.asarray([0.1, 0.9, 0.5])
y_col = []
for step in range(steps):
    y = y + dt * lorenz(y[0], y[1], y[2], s=s, r=r, b=b)
    y_col.append(y)
y_col = np.asarray(y_col)

# get inputs and targets
inputs = torch.tensor(y_col[:-lag, input_idx], device=device, dtype=dtype)
targets = torch.tensor(y_col[lag:], device=device, dtype=dtype)
for i in range(n_in):
    inputs[:, i] /= torch.max(inputs[:, i])
for i in range(n_out):
    targets[:, i] /= torch.max(targets[:, i])

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = RandomFeatureConceptorRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                                torch.tensor(W_z, device=device, dtype=dtype), lam, alphas[0])
rnn.init_new_conceptor(init_value="random")

# initial wash-out period
avg_input = torch.mean(inputs, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward_c(avg_input)

# train the conceptor
with torch.no_grad():
    for step in range(steps-lag):
        x = rnn.forward_c_adapt(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))

# harvest states
y_col = []
for step in range(loading_steps):
    rnn.forward_c(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))
    y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

# train readout
W_r, epsilon = rnn.train_readout(y_col.T, targets[:loading_steps].T, alphas[1])
print(f"Readout training error: {float(torch.mean(epsilon).cpu().detach().numpy())}")

# extract trained recurrent weights
c = rnn.C.cpu().detach().numpy().squeeze()
W = (rnn.W @ (torch.diag(rnn.C) @ rnn.W_z)).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(torch.diag(rnn.C) @ rnn.W_z)).cpu().detach().numpy())
print(f"Conceptor: {np.sum(c)}")

# generate predictions
with torch.no_grad():
    predictions = []
    y = W_r @ rnn.y
    for step in range(test_steps):
        y = W_r @ rnn.forward_c(y[input_idx])
        predictions.append(y.cpu().detach().numpy())
predictions = np.asarray(predictions)
targets = targets.cpu().detach().numpy()

# calculate wasserstein distance and get probability distributions
wd = 0.0
target_dist = []
prediction_dist = []
for i in range(n_out):
    wd_tmp, x_hist, y_hist, x_edges, y_edges = wasserstein(predictions[:, i], targets[:, i], n_bins=n_bins)
    wd += wd_tmp
    prediction_dist.append((x_edges, x_hist))
    target_dist.append((y_edges, y_hist))

# plotting
##########

# dynamics
fig, axes = plt.subplots(nrows=n_out, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.plot(targets[-plot_steps:, i], color="royalblue", label="target")
    ax.plot(predictions[-plot_steps:, i], color="darkorange", label="prediction")
    ax.set_ylabel(state_vars[i])
    if i == n_out-1:
        ax.set_xlabel("steps")
        ax.legend()
plt.tight_layout()

# trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Absolute weights: {np.round(W_abs, decimals=1)}")
plt.tight_layout()

# distributions
fig, axes = plt.subplots(nrows=n_out, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.bar(target_dist[i][0][:-1], target_dist[i][1], color="royalblue", label="target")
    ax.bar(prediction_dist[i][0][1:], prediction_dist[i][1], color="darkorange", label="prediction")
    ax.set_ylabel("p")
    ax.set_xlabel(state_vars[i])
    if i == n_out-1:
        ax.legend()
fig.suptitle(f"WD = {wd}")
plt.tight_layout()
plt.show()
