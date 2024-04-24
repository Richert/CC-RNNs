from src import RandomFeatureConceptorRNN
from src.functions import init_weights
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle


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
steps = 500000
init_steps = 1000

# reservoir parameters
N = 200
n_in = len(state_vars)
k = 600
sr = 0.99
bias_scale = 0.01
in_scale = 0.01
density = 0.1

# training parameters
backprop_steps = 2000
test_steps = 2000
scaling_steps = 4000
loading_steps = int(0.5*steps)
lr = 0.02
lam = 0.002
alpha = 50.0
betas = (0.9, 0.999)
tychinov = 1e-4

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
rnn = RandomFeatureConceptorRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                                torch.tensor(W_z, device=device, dtype=dtype), lam, alpha)
rnn.init_new_conceptor(init_value="random")

# initial wash-out period
avg_input = torch.mean(inputs, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward_c(avg_input)

# train the conceptor
with torch.no_grad():
    for step in range(steps-lag):
        x = rnn.forward_c_adapt(inputs[step])

# harvest states
y_col = []
for step in range(loading_steps):
    rnn.forward_c(inputs[step])
    y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

# load input into RNN weights
D, epsilon = rnn.load_input(y_col.T, inputs[1:loading_steps+1].T, tychinov)
print(f"Input loading error: {float(torch.mean(epsilon).cpu().detach().numpy())}")

# train readout
W_r, epsilon2 = rnn.train_readout(y_col.T, targets[:loading_steps].T, tychinov)
print(f"Readout training error: {float(torch.mean(epsilon2).cpu().detach().numpy())}")

# finalize conceptors
# with torch.no_grad():
#     rnn.y, rnn.z = y0, z0
#     for step in range(loading_steps):
#         rnn.forward_c_a_adapt()
c = rnn.C.cpu().detach().numpy()
print(f"Conceptor: {np.sum(c)}")

# generate predictions
with torch.no_grad():
    predictions = []
    for step in range(test_steps):
        y = rnn.forward_c_a()
        y = W_r @ y
        predictions.append(y.cpu().detach().numpy())
predictions = np.asarray(predictions)

# scaling task
with torch.no_grad():
    scaling_results = []
    c = rnn.C[:]
    scaling = torch.linspace(0.001, 10.0, scaling_steps)
    for step in range(scaling_steps):
        rnn.C = c*scaling[step]
        y = rnn.forward_c_a()
        y = W_r @ y
        scaling_results.append(y.cpu().detach().numpy())

# save results
results = {"targets": targets, "predictions": predictions, "frequency_scaling": scaling_results,
           "config": {"N": N, "k": k, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "lam": lam,
                      "alpha": alpha, "lag": lag}}
pickle.dump(results, open("../results/rfc_stuartlandau.pkl", "wb"))

# plotting
##########

# predictions
fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(state_vars[i])
    if i == n_in-1:
        ax.set_xlabel("steps")
        ax.legend()
plt.tight_layout()

# frequency scaling
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(n_in):
    ax.plot(targets[:plot_steps, i], label=f"y_{i+1}")
ax.set_xlabel("steps")
ax.legend()
plt.tight_layout()

plt.show()
