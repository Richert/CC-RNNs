import sys
sys.path.append('../')
from src import RandomFeatureConceptorRNN
import torch
import pickle
import numpy as np
from src.functions import init_weights
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

# batch condition
noise_lvl = float(sys.argv[-2])
rep = int(sys.argv[-1])

# general
dtype = torch.float64
device = "cpu"
plot_steps = 2000
state_vars = ["x", "y"]
lag = 1

# SL equation parameters
omega = 6.0
dt = 0.01

# reservoir parameters
N = 200
n_in = len(state_vars)
n_out = n_in
k = 10
sr = 0.99
bias_scale = 0.01
in_scale = 0.1
out_scale = 0.5
density = 0.2

# training parameters
steps = 500000
init_steps = 1000
backprop_steps = 5000
loading_steps = 100000
test_steps = 10000
lr = 0.008
lam = 0.002
betas = (0.9, 0.999)
alphas = (12.0, 1e-3, 1e-3)

# matrix initialization
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

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = RandomFeatureConceptorRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                                torch.tensor(W_z, device=device, dtype=dtype), lam, alphas[0])
rnn.free_param("W")
rnn.free_param("W_z")
rnn.init_new_conceptor(init_value="random")

# initial wash-out period
avg_input = torch.mean(inputs, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward_c(avg_input)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()), lr=lr, betas=betas)

# train the RNN weights and the conceptor simultaneously
current_loss = 0.0
loss_hist = []
with torch.enable_grad():

    loss = torch.zeros((1,))
    for step in range(steps-lag):

        # get RNN readout
        y = W_r @ rnn.forward_c_adapt(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))

        # calculate loss
        loss += loss_func(y, targets[step])

        # make update
        if (step + 1) % backprop_steps == 0:
            optim.zero_grad()
            W_z_tmp = torch.abs(rnn.W_z)
            for j in range(k):
                W_z_tmp[j, :] *= rnn.C[j]
            loss /= backprop_steps
            loss += alphas[1] * (torch.sum(torch.abs(rnn.W) @ W_z_tmp))
            loss.backward()
            current_loss = loss.item()
            loss_hist.append(current_loss)
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()

# store conceptor
rnn.store_conceptor("lorenz")
rnn.detach()

# harvest states
y_col = []
for step in range(loading_steps):
    rnn.forward_c(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))
    y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

# train readout
W_r, epsilon = rnn.train_readout(y_col.T, targets[:loading_steps].T, alphas[2])

# retrieve trained network connectivity
c = rnn.C.cpu().detach().numpy().squeeze()
W = (rnn.W @ (torch.diag(rnn.C) @ rnn.W_z)).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(torch.diag(rnn.C) @ rnn.W_z)).cpu().detach().numpy())

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

# save results
results = {"targets": targets[loading_steps:loading_steps+test_steps], "predictions": predictions,
           "config": {"N": N, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "k": k, "alphas": alphas},
           "condition": {"repetition": rep, "noise": noise_lvl},
           "training_error": epsilon, "avg_weights": W_abs, "wd": wd, "loss_hist": loss_hist}
pickle.dump(results, open(f"../results/clr/stuartlandau_noise{int(noise_lvl*100)}_{rep}.pkl", "wb"))
