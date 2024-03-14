from src import LowRankRNN
import torch
import matplotlib.pyplot as plt
import numpy as np


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


def minmax(x: np.ndarray) -> np.ndarray:
    x = x - np.min(x)
    x = x / np.max(x)
    return x


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"

# lorenz equation parameters
s = 20.0
r = 28.0
b = 2.667
dt = 0.01
steps = 200000
init_steps = 1000

# reservoir parameters
N = 500
n_in = 3
k = 1
sr = 0.99
bias = 0.9
in_scale = 1.2
density = 0.5

# training parameters
backprop_steps = 500
test_steps = 2000
lr = 0.1
betas = (0.9, 0.999)

# generate inputs and targets
#############################

# simulation
y = np.asarray([0.1, 0.9, 1.1])
y_col = []
for step in range(steps + init_steps):
    y = y + dt * lorenz(y[0], y[1], y[2], s=s, r=r, b=b)
    y_col.append(y)
y_col = np.asarray(y_col)

# plotting
plot_steps = 2000
lorenz_vars = ["x", "y", "z"]
fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
for i, (ax) in enumerate(axes):
    ax.plot(y_col[init_steps:init_steps+plot_steps, i])
    if i == 2:
        ax.set_xlabel("steps")
plt.suptitle("Lorenz dynamics")
plt.tight_layout()

# get inputs and targets
inputs = torch.tensor(y_col[:-1], device=device, dtype=dtype)
targets = torch.tensor(y_col[1:], device=device, dtype=dtype)

# train low-rank RNN to predict next time step of Lorenz attractor
##################################################################

# initialize LR-RNN
rnn = LowRankRNN(N, n_in, k, sr=sr, density=density, bias_var=bias, rf_var=in_scale, device=device, dtype=dtype)
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
    for step in range(steps-1):

        # get RNN output
        x = rnn.forward(inputs[step])
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

# generate predictions
predictions = []
for step in range(test_steps):

    # get RNN output
    x = rnn.forward(inputs[step])
    y = readout.forward(x)

    # store results
    predictions.append(y.cpu().detach().numpy())

predictions = np.asarray(predictions)

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 6))

for i, ax in enumerate(axes):

    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    if i == 2:
        ax.set_xlabel("steps")
        ax.legend()

plt.tight_layout()
plt.show()
