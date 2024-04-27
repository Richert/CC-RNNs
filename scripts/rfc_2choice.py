from src import RandomFeatureConceptorRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights
import pickle


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


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 4000

# input parameters
in_vars = ["x", "y"]
n_epochs = 200
epoch_steps = 1000
signal_scale = 1.0
noise_scale = 0.1
steps = int(n_epochs*epoch_steps)

# reservoir parameters
N = 200
n_in = len(in_vars)
k = 400
sr = 0.99
bias_scale = 0.01
in_scale = 0.01
density = 0.1

# matrix initialization
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)

# training parameters
test_steps = 10000
loading_steps = int(0.5 * steps)
lam = 0.002
alpha = 4.0
betas = (0.9, 0.999)
tychinov = 1e-3

# generate inputs and targets
#############################

inputs, targets = [], []
for n in range(n_epochs):

    mus = signal_scale * torch.randn((n_in,), device=device, dtype=dtype)
    inp = torch.zeros((epoch_steps, n_in), device=device, dtype=dtype)
    for i, mu in enumerate(mus):
        inp[:, i] += mu + noise_scale * torch.randn((epoch_steps,), device=device, dtype=dtype)
    targ = torch.zeros_like(inp)
    targ[:, torch.argmax(mus)] = 1.0
    inputs.append(inp)
    targets.append(targ)

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = RandomFeatureConceptorRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                                torch.tensor(W_z, device=device, dtype=dtype), lam, alpha)
rnn.init_new_conceptor(init_value="random")

# initial wash-out period
avg_input = torch.mean(inputs[0], dim=0)
with torch.no_grad():
    for step in range(epoch_steps):
        rnn.forward_c(avg_input)

# train the conceptor
with torch.no_grad():
    for epoch in range(n_epochs):
        for step in range(epoch_steps):
            x = rnn.forward_c_adapt(inputs[epoch][step])

# harvest states
y_col = []
with torch.no_grad():
    for epoch in range(n_epochs):
        for step in range(epoch_steps):
            rnn.forward_c(inputs[epoch][step])
            y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

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
    y = W_r @ rnn.y
    for step in range(test_steps):
        y = W_r @ rnn.forward_c(y)
        predictions.append(y.cpu().detach().numpy())
predictions = np.asarray(predictions)

# save results
# results = {"targets": targets, "predictions": predictions,
#            "config": {"N": N, "k": k, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "lam": lam,
#                       "alpha": alpha, "lag": lag}}
# pickle.dump(results, open("../results/rfc_lorenz.pkl", "wb"))

# plotting
##########

fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))

for i, ax in enumerate(axes):

    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(state_vars[i])
    if i == n_in-1:
        ax.set_xlabel("steps")
        ax.legend()

plt.tight_layout()
plt.show()
