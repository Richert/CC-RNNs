from src import ConceptorLowRankRNN
from src.functions import init_weights
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import wasserstein_distance


# function definitions
######################

def stuart_landau(x: float, y: float, omega: float = 10.0) -> np.ndarray:
    """
    Parameters
    ----------
    x, y: float
        State variables.
    omega : float
       Parameters defining the Stuart-Landau attractor.

    Returns
    -------
    xy_dot : np.ndarray
       Vectorfield of the Lorenz equations.
    """
    x_dot = omega*y + y*(1-y**2-x**2)
    y_dot = -omega*x + y*(1-y**2-x**2)
    return np.asarray([x_dot, y_dot])


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 2000
state_vars = ["x", "y"]
lag = 1

# SL equation parameters
omega = 6.0
dt = 0.01
noise_lvl = 0.8

# reservoir parameters
N = 100
n_in = len(state_vars)
k = 250
sr = 1.1
bias_scale = 0.01
in_scale = 0.1
density = 0.1

# training parameters
steps = 500000
test_steps = 10000
init_steps = 1000
loading_steps = 100000
lam = 0.002
alphas = (10.0, 1e-3)
betas = (0.9, 0.999)

# matrix initialization
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)

# generate inputs and targets
#############################

# simulation
y = np.asarray([0.1, 0.9])
y_col = []
for step in range(steps):
    y = y + dt * stuart_landau(y[0], y[1], omega=omega)
    y_col.append(y)
y_col = np.asarray(y_col)

# get inputs and targets
inputs = torch.tensor(y_col[:-lag], device=device, dtype=dtype)
targets = torch.tensor(y_col[lag:], device=device, dtype=dtype)

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = ConceptorLowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
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

# retrieve network connectivity
c = rnn.c_weights.cpu().detach().numpy().squeeze()
W = (rnn.L @ (torch.diag(rnn.c_weights) @ rnn.L)).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(torch.diag(rnn.c_weights) @ rnn.L)).cpu().detach().numpy())
print(f"Conceptor: {np.sum(c)}")

# generate predictions
with torch.no_grad():
    predictions = []
    y = W_r @ rnn.y
    for step in range(test_steps):
        y = W_r @ rnn.forward_c(y)
        predictions.append(y.cpu().detach().numpy())
predictions = np.asarray(predictions)
targets = targets.cpu().detach().numpy()

# calculate prediction error
cutoff = 1
f0, p0 = welch(targets[cutoff:, 0], fs=10/dt, nperseg=2048)
f1, p1 = welch(predictions[cutoff:, 0], fs=10/dt, nperseg=2048)
p0 /= np.sum(p0)
p1 /= np.sum(p1)
wd = wasserstein_distance(u_values=f0, v_values=f1, u_weights=p0, v_weights=p1)

# plotting
##########

# dynamics
fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(state_vars[i])
    if i == n_in-1:
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
plt.show()
