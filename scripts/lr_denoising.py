from src import LowRankRNN
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
omega = 6.0
dt = 0.01
steps = 1000000
init_steps = 1000
noise_lvl = 0.3
lag = 15

# reservoir parameters
N = 200
n_in = len(state_vars)
k = 1
sr = 1.2
bias_scale = 0.9
in_scale = 1.2
density = 0.2

# rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)

# training parameters
backprop_steps = 2000
test_steps = 2000
lr = 0.1
betas = (0.9, 0.999)
tychinov_alpha = 1e-3

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
        x = rnn.forward(inputs[step] + torch.randn(size=(n_in,), device=device, dtype=dtype)*noise_lvl)
        y = readout.forward(x)

        # calculate loss
        loss += loss_func(y, targets[step])

        # make update
        if (step + 1) % backprop_steps == 0:

            optim.zero_grad()
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            print(f"Current loss: {current_loss}")

# load input pattern into RNN weights and generate predictions
##############################################################

y0 = rnn.y[:]

# load input pattern into RNN
optim = torch.optim.Adam(list(readout.parameters()), lr=0.01, betas=betas)
y_col = []

for step in range(steps-lag):

    y_col.append(rnn.y)

    # get RNN output
    x = rnn.forward(inputs[step] + torch.randn(size=(n_in,), device=device, dtype=dtype)*noise_lvl)
    y = readout.forward(x)

    # calculate loss
    loss += loss_func(y, targets[step])

    # make update
    if (step + 1) % backprop_steps == 0:
        optim.zero_grad()
        loss.backward()
        current_loss = loss.item()
        optim.step()
        loss = torch.zeros((1,))
        print(f"Readout loss: {current_loss}")

# generate predictions
with torch.no_grad():
    rnn.y = y0
    predictions = []
    input_col = []
    y = readout.forward(rnn.y)
    for step in range(test_steps):

        # get RNN output
        inp = inputs[step] + torch.randn(size=(n_in,), device=device, dtype=dtype)*noise_lvl
        x = rnn.forward(inp)
        y = readout.forward(x)

        # store results
        predictions.append(y.cpu().detach().numpy())
        input_col.append(inp.cpu().detach().numpy())

predictions = np.asarray(predictions)
inputs = np.asarray(input_col)

# plotting
##########

fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))

for i, ax in enumerate(axes):

    ax.plot(inputs[:plot_steps, i], color="black", label="input")
    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(state_vars[i])
    if i == n_in-1:
        ax.set_xlabel("steps")
        ax.legend()

plt.tight_layout()
plt.show()

# save the network configuration
################################

# A = rnn.A.detach().cpu().numpy()
# B = rnn.B.detach().cpu().numpy()
# W_in = rnn.W_in.detach().cpu().numpy()
# bias = rnn.bias.detach().cpu().numpy()
# inputs = inputs.detach().cpu().numpy()
# targets = targets.detach().cpu().numpy()
# pickle.dump({"omega": omega, "A": A, "B": B, "W_in": W_in, "bias": bias, "inputs": inputs, "targets": targets},
#             open(f"../results/lr_stuartlandau_{int(omega)}Hz.pkl", "wb"))
