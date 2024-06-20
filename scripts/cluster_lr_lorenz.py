import sys
sys.path.append('../')
from src import LowRankRNN
import torch
import numpy as np
from src.functions import init_weights
from scipy.stats import wasserstein_distance
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

# batch condition
noise_lvl = float(sys.argv[-2])
rep = int(sys.argv[-1])

# general
dtype = torch.float64
device = "cpu"
state_vars = ["x", "y", "z"]
lag = 1
n_bins = 500

# lorenz equation parameters
s = 10.0
r = 28.0
b = 8/3
dt = 0.01
input_idx = np.asarray([0, 1, 2])

# reservoir parameters
N = 200
n_in = len(input_idx)
n_out = len(state_vars)
k = len(state_vars)
sr = 0.99
bias_scale = 0.01
in_scale = 0.1
out_scale = 0.5
density = 0.2

# rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_out, N), device=device, dtype=dtype)

# training parameters
steps = 500000
init_steps = 1000
backprop_steps = 5000
loading_steps = 100000
test_steps = 10000
lr = 0.008
betas = (0.9, 0.999)
alphas = (1e-3, 1e-3)

# generate inputs and targets
#############################

# simulation
y = np.asarray([0.1, 0.9, 1.1])
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

# train low-rank RNN to predict next time step of Lorenz attractor
##################################################################

# initialize LR-RNN
rnn = LowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias, torch.tensor(W_z, device=device, dtype=dtype))
rnn.free_param("W")
rnn.free_param("W_z")

# initial wash-out period
avg_input = torch.mean(inputs, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()), lr=lr, betas=betas)

# training
current_loss = 0.0
loss_hist = []
with torch.enable_grad():

    loss = torch.zeros((1,))
    for step in range(steps-lag):

        # get RNN output
        y = W_r @ rnn.forward(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))

        # calculate loss
        loss += loss_func(y, targets[step])

        # make update
        if (step + 1) % backprop_steps == 0:

            optim.zero_grad()
            loss /= backprop_steps
            loss += alphas[0] * torch.sum(torch.abs(rnn.W) @ torch.abs(rnn.W_z))
            loss.backward()
            current_loss = loss.item()
            loss_hist.append(current_loss)
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()

W = (rnn.W @ rnn.W_z).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(rnn.W_z)).cpu().detach().numpy())

# train final readout and generate predictions
##############################################

# harvest states
y_col = []
with torch.no_grad():
    for step in range(loading_steps):
        y = rnn.forward(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))
        y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

# train readout
W_r, epsilon = rnn.train_readout(y_col.T, targets[:loading_steps].T, alphas[1])
epsilon = float(torch.mean(epsilon).cpu().detach().numpy())

# generate predictions
with torch.no_grad():
    predictions = []
    y = W_r @ rnn.y
    for step in range(test_steps):
        y = W_r @ rnn.forward(y[input_idx])
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

# save results
results = {"targets": targets[loading_steps:loading_steps+test_steps], "predictions": predictions,
           "config": {"N": N, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "k": k, "alphas": alphas},
           "condition": {"noise": noise_lvl, "repetition": rep},
           "training_error": epsilon, "avg_weights": W_abs, "loss_hist": loss_hist,
           "prediction_dist": prediction_dist, "target_dist": target_dist, "wd": wd}
pickle.dump(results, open(f"../results/lr/lorenz_noise{int(noise_lvl*100)}_{rep}.pkl", "wb"))
