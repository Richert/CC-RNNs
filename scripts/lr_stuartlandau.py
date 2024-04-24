from src import LowRankRNN
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
k = 1
sr = 1.1
bias_scale = 0.01
in_scale = 0.1
out_scale = 0.5
density = 0.2

# training parameters
backprop_steps = 500
test_steps = 2000
scaling_steps = 4000
loading_steps = int(0.5*steps)
lr = 0.05
betas = (0.9, 0.999)
tychinov_alpha = 1e-3

# rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_in, N), device=device, dtype=dtype)

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

# train low-rank RNN to predict next time step of Lorenz attractor
##################################################################

# initialize LR-RNN
rnn = LowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias, torch.tensor(W_z, device=device, dtype=dtype))
readout = torch.nn.Linear(N, n_in, bias=True, device=device, dtype=dtype)

# initial wash-out period
avg_input = torch.mean(inputs, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + list(readout.parameters()), lr=lr, betas=betas)

# training
current_loss = 0.0
with torch.enable_grad():

    loss = torch.zeros((1,))
    for step in range(steps-lag):

        # get RNN output
        x = rnn.forward(inputs[step])
        y = readout.forward(x)

        # calculate loss
        loss += loss_func(y, targets[step])

        # make update
        if (step + 1) % backprop_steps == 0:

            optim.zero_grad()
            loss /= backprop_steps
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            print(f"Training phase I loss: {current_loss}")

# load input pattern into RNN weights and generate predictions
##############################################################

# load input pattern into RNN
optim = torch.optim.Adam(list(readout.parameters()), lr=lr, betas=betas)
y_col = []

for step in range(loading_steps):

    y_col.append(rnn.y)

    # get RNN output
    x = rnn.forward(inputs[step])
    y = readout.forward(x)

    # calculate loss
    loss += loss_func(y, targets[step])

    # make update
    if (step + 1) % backprop_steps == 0:
        optim.zero_grad()
        loss /= backprop_steps
        loss.backward()
        current_loss = loss.item()
        optim.step()
        loss = torch.zeros((1,))
        print(f"Training phase II loss: {current_loss}")

# compute input simulation weight matrix
D, epsilon = rnn.load_input(torch.stack(y_col, dim=0).T, inputs[:loading_steps].T, tychinov_alpha)
print(f"Input loading error: {float(torch.mean(epsilon).cpu().detach().numpy())}")

# generate predictions
drive_steps = 500
with torch.no_grad():
    predictions = []
    for step in range(test_steps):

        # get RNN output
        if step < drive_steps:
            x = rnn.forward(inputs[step])
        else:
            x = rnn.forward_a()
        y = readout.forward(x)

        # store results
        predictions.append(y.cpu().detach().numpy())

predictions = np.asarray(predictions)

# scaling task
with torch.no_grad():
    scaling_results = []
    W, W_z = rnn.W[:], rnn.W_z[:]
    scaling = torch.linspace(0.1, 10.0, scaling_steps)
    for step in range(scaling_steps):
        rnn.W = W*scaling[step]
        rnn.W_z = W_z*scaling[step]
        x = rnn.forward_a()
        y = readout.forward(x)
        scaling_results.append(y.cpu().detach().numpy())

scaling_results = np.asarray(scaling_results)

# save results
results = {"targets": targets, "predictions": predictions, "freqency_scaling": scaling_results,
           "config": {"N": N, "k": k, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "lag": lag}}
pickle.dump(results, open("../results/lr_stuartlandau.pkl", "wb"))

# plotting
##########

# predictions
fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))

for i, ax in enumerate(axes):

    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.axvline(x=drive_steps, color="black", linestyle="dashed")
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
