from src import RandomFeatureConceptorRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
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
interp_steps = 4000
state_vars = ["x", "y"]

# SL equation parameters
omegas = [3, 12]
dt = 0.01
steps = 1000000
init_steps = 1000

# rnn parameters
N = 200
n_in = len(state_vars)
k = 20
sr = 1.4
bias_scale = 1.1
in_scale = 1.2
out_scale = 1.0
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
backprop_steps = 400
test_steps = 2000
loading_steps = int(0.5 * (steps - 1))
lr = 0.01
lam = 2e-3
alpha = 2.0
betas = (0.9, 0.999)
tychinov = 1e-3
epsilon = 1e-5

# train LR-RNN weights
######################

# initialize RFC
rnn = RandomFeatureConceptorRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                                torch.tensor(W_z, device=device, dtype=dtype), lam, alpha)
rnn.free_param("W")
rnn.free_param("W_z")

# set up loss function
loss_func = torch.nn.MSELoss()

target_col, input_col, init_states = {}, {}, {}
for i, omega in enumerate(omegas):

    # set up optimizer
    optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

    # generate inputs and targets
    y = np.asarray([0.1, 0.9])
    y_col = []
    for step in range(steps):
        y = y + dt * stuart_landau(y[0], y[1], omega=omega)
        y_col.append(y)
    y_col = np.asarray(y_col)
    inputs = torch.tensor(y_col[:-1], device=device, dtype=dtype)
    targets = torch.tensor(y_col[1:], device=device, dtype=dtype)
    target_col[omega] = targets[:loading_steps]
    input_col[omega] = inputs[:loading_steps]

    # initialize new conceptor
    rnn.init_new_conceptor(init_value="random")
    if i > 0:
        rnn.C = rnn.combine_conceptors(rnn.C, 1 - rnn.conceptors[omegas[i-1]], operation="and")

    # initial wash-out period
    avg_input = torch.mean(inputs, dim=0)
    with torch.no_grad():
        for step in range(init_steps):
            rnn.forward(avg_input)

    # train the RNN weights and the conceptor simultaneously
    current_loss = 0.0
    with torch.enable_grad():

        loss = torch.zeros((1,))
        for step in range(steps-1):

            # get RNN readout
            y = W_r @ rnn.forward_c_adapt(inputs[step])

            # calculate loss
            loss += loss_func(y, targets[step])

            # make update
            if (step + 1) % backprop_steps == 0:
                optim.zero_grad()
                W_r_tmp = torch.abs(rnn.W_z)
                for j in range(k):
                    W_r_tmp[j, :] *= rnn.C[j]
                loss += epsilon*torch.sum(torch.abs(rnn.W) @ W_r_tmp)
                loss.backward()
                current_loss = loss.item()
                optim.step()
                loss = torch.zeros((1,))
                rnn.detach()
                print(f"Training phase I loss (omega = {omega}): {current_loss}")

    # store final state
    rnn.store_conceptor(omega)
    rnn.detach()
    init_states[omega] = (rnn.y[:], rnn.z[:])

# finalize conceptors
for i, omega in enumerate(omegas):
    c1, c2 = rnn.conceptors[omega], rnn.conceptors[omegas[1 - i]]
    rnn.conceptors[omega] = rnn.combine_conceptors(c1, 1 - c2, operation="and")

# train joint readout
#####################

state_col = {}
with torch.no_grad():

    for omega in omegas:

        # apply condition
        inputs = input_col[omega]
        rnn.activate_conceptor(omega)
        rnn.y, rnn.z = init_states[omega]

        # harvest states
        y_col = []
        for step in range(loading_steps):
            y_col.append(rnn.y)
            rnn.forward_c(inputs[step])
        state_col[omega] = torch.stack(y_col, dim=0)

    # train readout
    states = torch.cat([state_col[omega][1:] for omega in omegas], dim=0)
    targets = torch.cat([target_col[omega][:-1] for omega in omegas], dim=0)
    W_r, epsilon2 = rnn.train_readout(states.T, targets.T, tychinov)
    print(f"Readout training error: {float(torch.mean(epsilon2).cpu().detach().numpy())}")

# generate predictions
######################

# single frequency oscillation
prediction_col = {}
for i, omega in enumerate(omegas):

    rnn.activate_conceptor(omega)
    c = rnn.conceptors[omega].detach().cpu().numpy()
    targets = target_col[omega]
    inputs = input_col[omega]
    print(f"Conceptor for omega = {omega}: {np.round(c, decimals=2)}")

    # generate prediction
    with torch.no_grad():

        rnn.y, rnn.z = init_states[omega]
        predictions = []
        for step in range(test_steps):

            # get RNN readout
            y = W_r @ rnn.forward_c(inputs[step])

            # store results
            predictions.append(y.cpu().detach().numpy())

    prediction_col[omega] = np.asarray(predictions)
    target_col[omega] = targets.detach().cpu().numpy()

# linear interpolation between two conceptors for each input frequency
interp_col = {}
for omega in omegas:

    inputs = input_col[omega]

    c1 = rnn.conceptors[omegas[0]]
    c2 = rnn.conceptors[omegas[1]]

    constant_steps = int(interp_steps / 4)
    ramp_steps = interp_steps - constant_steps
    gamma = torch.zeros((interp_steps,), dtype=dtype, device=device)
    gamma[constant_steps:ramp_steps] = torch.linspace(0.0, 1.0, steps=ramp_steps - constant_steps)
    gamma[ramp_steps:] = 1.0

    interp_tmp = []
    with torch.no_grad():
        rnn.y, rnn.z = init_states[omega]
        for step in range(interp_steps):
            rnn.C = gamma[step]*c2 + (1-gamma[step])*c1
            y = W_r @ rnn.forward_c(inputs[step])
            interp_tmp.append(y.cpu().detach().numpy())
    interp_tmp = np.asarray(interp_tmp)
    interp_col[omega] = interp_tmp

# plotting
##########

# prediction figure
fig, axes = plt.subplots(ncols=n_in, nrows=len(omegas), figsize=(12, 6))
for i in range(len(omegas)):
    for j in range(n_in):

        ax = axes[i, j]
        ax.plot(target_col[omegas[i]][:plot_steps, j], color="royalblue", label="target")
        ax.plot(prediction_col[omegas[i]][:plot_steps, j], color="darkorange", label="prediction")
        ax.set_ylabel(state_vars[j])
        if i == len(omegas)-1:
            ax.set_xlabel("steps")
            ax.legend()

fig.suptitle("Model Predictions")
plt.tight_layout()
plt.show()

# interpolation figure
fig, axes = plt.subplots(ncols=n_in, nrows=len(omegas), figsize=(12, 6))
for i in range(len(omegas)):
    for j in range(n_in):

        ax = axes[i, j]
        ax.plot(target_col[omegas[i]][:interp_steps, j], color="royalblue", label="target")
        ax.plot(interp_col[omegas[i]][:interp_steps, j], color="darkorange", label="prediction")
        ax.set_ylabel(state_vars[j])
        if i == len(omegas)-1:
            ax.set_xlabel("steps")
            ax.legend()

fig.suptitle("Conceptor interpolations")
plt.tight_layout()
plt.show()
