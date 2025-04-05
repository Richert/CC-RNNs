from src.rnn import LowRankCRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights


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

# SL equation parameters
omegas = [4, 6]
dt = 0.01
steps = 200000
init_steps = 1000
lag = 5

# rnn parameters
N = 200
n_in = len(state_vars)
k = len(omegas)
sr = 0.99
bias_scale = 0.01
in_scale = 1.0
out_scale = 0.1
density = 0.5

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_in, N), device=device, dtype=dtype)

# training parameters
backprop_steps = 1000
test_steps = 2000
loading_steps = int(0.5 * (steps - 1))
lr = 0.05
lam = 0.002
alpha = 5.0
betas = (0.9, 0.999)
tychinov = 1e-3

# conceptors
conceptors = {}
for i, omega in enumerate(omegas):
    c = torch.zeros((k,), device=device, dtype=dtype)
    c[i] = 1.0
    conceptors[omega] = c

# train LR-RNN weights
######################

# initialize RFC
rnn = LowRankCRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                  torch.tensor(W_z, device=device, dtype=dtype), alpha=alpha, lam=lam)
rnn.conceptors.update(conceptors)
rnn.free_param("W")
rnn.free_param("W_z")

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(rnn.parameters(), lr=lr, betas=betas)

target_col, input_col, init_states = {}, {}, {}
for omega in omegas:

    # impose conceptor
    rnn.activate_conceptor(omega)

    # generate inputs and targets
    y = np.asarray([0.1, 0.9])
    y_col = []
    for step in range(steps):
        y = y + dt * stuart_landau(y[0], y[1], omega=omega)
        y_col.append(y)
    y_col = np.asarray(y_col)
    inputs = torch.tensor(y_col[:-lag], device=device, dtype=dtype)
    targets = torch.tensor(y_col[lag:], device=device, dtype=dtype)
    target_col[omega] = targets[:loading_steps]
    input_col[omega] = inputs[:loading_steps]

    # initial wash-out period
    avg_input = torch.mean(inputs, dim=0)
    with torch.no_grad():
        for step in range(init_steps):
            rnn.forward_c(avg_input)

    # training the RNN weights
    current_loss = 0.0
    with torch.enable_grad():

        loss = torch.zeros((1,))
        for step in range(steps-lag):

            # get RNN readout
            y = W_r @ rnn.forward_c(inputs[step])

            # calculate loss
            loss += loss_func(y, targets[step])

            # make update
            if (step + 1) % backprop_steps == 0:
                optim.zero_grad()
                loss.backward()
                current_loss = loss.item()
                optim.step()
                loss = torch.zeros((1,))
                rnn.detach()
                print(f"Training phase I loss (omega = {omega}): {current_loss}")

    # store final state
    rnn.detach()
    # init_states[omega] = rnn.y[:]

# load input into RNN weights and train readout
###############################################

state_col = {}
with torch.no_grad():

    for omega in omegas:

        # apply condition
        inputs = input_col[omega]
        targets = target_col[omega]
        # rnn.y = init_states[omega]
        rnn.activate_conceptor(omega)

        # harvest states
        y_col = []
        for step in range(loading_steps):
            rnn.forward_c(inputs[step])
            y_col.append(rnn.y)

        state_col[omega] = torch.stack(y_col, dim=0)
        rnn.detach()

    # load input into RNN weights
    inputs = torch.cat([input_col[omega][1:] for omega in omegas], dim=0)
    states = torch.cat([state_col[omega][:-1] for omega in omegas], dim=0)
    D, epsilon = rnn.load_input(states.T, inputs.T, tychinov, overwrite=True)
    print(f"Input loading error: {float(torch.mean(epsilon).cpu().detach().numpy())}")

    # train readout
    states = torch.cat([state_col[omega] for omega in omegas], dim=0)
    targets = torch.cat([target_col[omega] for omega in omegas], dim=0)
    W_r, epsilon2 = rnn.train_readout(states.T, targets.T, tychinov)
    print(f"Readout training error: {float(torch.mean(epsilon2).cpu().detach().numpy())}")

# generate predictions
######################

# single frequency oscillation
prediction_col = {}
for omega in omegas:

    rnn.activate_conceptor(omega)
    target = target_col[omega]

    # generate prediction
    with torch.no_grad():

        # rnn.y = init_states[omega]
        predictions = []
        for step in range(test_steps):

            # get RNN readout
            y = W_r @ rnn.forward_c_a()

            # store results
            predictions.append(y.cpu().detach().numpy())

    prediction_col[omega] = np.asarray(predictions)
    target_col[omega] = target.detach().cpu().numpy()

# linear interpolation between two frequencies
c1, c2 = rnn.conceptors[omegas[0]], rnn.conceptors[omegas[1]]
interpolation_steps = 4000
constant_steps = int(interpolation_steps/4)
ramp_steps = interpolation_steps - constant_steps
gamma = torch.zeros((interpolation_steps,), dtype=dtype, device=device)
gamma[constant_steps:ramp_steps] = torch.linspace(0.0, 1.0, steps=ramp_steps - constant_steps)
gamma[ramp_steps:] = 1.0
interp_col = []
with torch.no_grad():
    for step in range(interpolation_steps):
        rnn.c_weights = gamma[step] * c1 + (1 - gamma[step]) * c2
        y = W_r @ rnn.forward_c_a()
        interp_col.append(y.cpu().detach().numpy())
interp_col = np.asarray(interp_col)

# plotting
##########

# prediction figure
fig, axes = plt.subplots(ncols=n_in, nrows=k, figsize=(12, 6))
for i in range(k):
    for j in range(n_in):

        ax = axes[i, j]
        ax.plot(target_col[omegas[i]][:plot_steps, j], color="royalblue", label="target")
        ax.plot(prediction_col[omegas[i]][:plot_steps, j], color="darkorange", label="prediction")
        ax.set_ylabel(state_vars[j])
        if i == k-1:
            ax.set_xlabel("steps")
            ax.legend()

plt.tight_layout()
fig.suptitle("Model Predictions")

# interpolation figure
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(interp_col[:, 0], label="x")
ax2.plot(interp_col[:, 1], label="y")
ax2.legend()
ax2.set_xlabel("steps")
fig2.suptitle("Frequency interpolation")
plt.tight_layout()
plt.show()
