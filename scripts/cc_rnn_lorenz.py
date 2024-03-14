from src import RandomFeatureConceptorRNN
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
N = 200
n_in = 3
k = 1000
sr = 0.99
bias = 0.9
in_scale = 1.2
density = 0.5

# training parameters
backprop_steps = 500
test_steps = 2000
lr = 0.01
alpha = 200.0

# generate inputs and targets
#############################

# simulation
y = np.asarray([0.1, 0.9, 1.1])
y_col = []
for step in range(steps + init_steps):
    y = y + dt * lorenz(y[0], y[1], y[2], s=s, r=r, b=b)
    y_col.append(y)
y_col = np.asarray(y_col)

# get inputs and targets
inputs = torch.tensor(y_col[:-1], device=device, dtype=dtype)
targets = torch.tensor(y_col[1:], device=device, dtype=dtype)

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = RandomFeatureConceptorRNN(N, n_in, k, lam=lr, alpha=alpha, sr=sr, density=density, bias_var=bias, rf_var=in_scale,
                                device=device, dtype=dtype)
readout = torch.nn.Linear(N, n_in, bias=True, device=device, dtype=dtype)

# initial wash-out period
avg_input = torch.mean(inputs, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)

# training
current_loss = 0.0
with torch.enable_grad():

    loss = torch.zeros((1,))
    for step in range(steps-1):

        # get RNN output
        x = rnn.forward_c_adapt(inputs[step])
        y = readout.forward(x)
