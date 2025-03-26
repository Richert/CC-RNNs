from src.rnn import HHRNN
from src.functions import init_weights, init_dendrites
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# function definitions
######################

def vanderpol(y: torch.Tensor, x: torch.Tensor, mu: float, alpha: float = 0.2) -> torch.Tensor:
    y1_dot = alpha*y[1]
    y2_dot = alpha*(y[1]*(mu + x)*(1 - y[0]**2) - y[0])
    return torch.stack([y1_dot, y2_dot], dim=0)

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 2000
state_vars = ["x", "y"]

# task parameters
min_mu, max_mu = -0.5, 0.5
noise_lvl = 0.6
sigma = 100
d = 1
dt = 1e-2
steps = 10000
init_steps = 1000
truncation_steps = 1000

# HH parameters
gamma = 10.0
theta = 40.0

# rnn parameters
n_in = len(state_vars)
n_out = len(state_vars)
k = 10
n_dendrites = 10
N = int(k*n_dendrites)
sr = 0.99
bias_scale = 0.1
in_scale = 1.0
out_scale = 0.02
density = 0.5

# training parameters
trials = 100
train_trials = int(0.9*trials)
test_trials = trials - train_trials
lr = 1e-3
betas = (0.9, 0.999)
batch_size = 10
gradient_cutoff = 1e4

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
L = np.abs(init_weights(N, k, density))
R = init_dendrites(k, n_dendrites)
W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

# plot coupling matrices
# fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
# ax = axes[0]
# ax.imshow(L, aspect="auto", interpolation="none")
# ax.set_xlabel("from: neurons")
# ax.set_ylabel("to: dendrites")
# ax = axes[1]
# ax.imshow(R, aspect="auto", interpolation="none")
# ax.set_xlabel("from: dendrites")
# ax.set_ylabel("to: neurons")
# plt.tight_layout()
# plt.show()

# generate input and targets
############################

# generate targets and inputs
y0 = torch.zeros((2,), device=device, dtype=dtype)
y0[0] = 1e-1
targets, inputs = [], []
with torch.no_grad():
    for n in range(trials):
        successful = False
        while not successful:
            mu = (max_mu - min_mu) * np.random.rand() + min_mu
            y = y0.detach()
            x = gaussian_filter1d(np.random.randn(steps + d), sigma=sigma)
            x *= noise_lvl / np.max(x)
            y_col = []
            for step in range(steps + d):
                y = y + dt * vanderpol(y, x=x[step], mu=mu)
                y_col.append(y)
            y_col = np.asarray(y_col)
            if np.isfinite(y_col[-1, 0]):
                successful = True
        inputs.append(y_col[:-d])
        targets.append(y_col[d:])

# plot targets
# n_examples = 3
# fig, axes = plt.subplots(nrows=n_examples, figsize=(12, 3*n_examples))
# for i in range(n_examples):
#     ax = axes[i]
#     ax.plot(targets[i][:, 0], label="x")
#     ax.plot(targets[i][:, 1], label="y")
#     ax.set_xlabel("time")
#     ax.set_ylabel("amplitude")
#     ax.legend()
# fig.suptitle("Target waveforms")
# plt.tight_layout()
# plt.show()

# train LR-RNN weights
######################

# initialize RFC
rnn = HHRNN(torch.tensor(L, dtype=dtype, device=device), torch.tensor(R, device=device, dtype=dtype), W_in, bias, dt=dt,
            theta=theta, gamma=gamma)
rnn.free_param("W")
rnn.free_param("W_z")
rnn.free_param("W_in")
rnn.free_param("bias")

# initial wash-out period
avg_input = torch.zeros((n_in,), device=device, dtype=dtype)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)
init_state = (v[:] for v in rnn.state_vars)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
params = list(rnn.parameters()) + [W_r]
optim = torch.optim.Adam(params, lr=lr, betas=betas)
rnn.clip(gradient_cutoff)

# training
loss = torch.zeros((1,))
current_loss = 0.0
torch.autograd.set_detect_anomaly(True)
with torch.enable_grad():

    for trial in range(train_trials):

        # get input and target timeseries
        inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
        target = torch.tensor(targets[trial], device=device, dtype=dtype)

        # initial condition
        rnn.detach()
        rnn.set_state(init_state)

        # collect loss
        y = y0.detach()
        y_col = []
        for step in range(steps):
            rnn.forward(inp[step])
            y = W_r @ rnn.z
            if step % truncation_steps == truncation_steps - 1:
                rnn.detach()
            y_col.append(y)

        # calculate loss
        y_col = torch.stack(y_col, dim=0)
        loss += loss_func(y_col, target)

        # make update
        if trial % batch_size == batch_size - 1:
            optim.zero_grad()
            loss.backward()
            optim.step()
            current_loss = loss.item()
            loss = torch.zeros((1,))
            print(f"Training trial {trial + 1} / {train_trials}: MSE = {current_loss}")

# generate predictions
######################

predictions = []
model_dynamics = []
with torch.no_grad():
    for trial in range(train_trials, trials):

        # get input and target timeseries
        inp = torch.tensor(inputs[trial], device=device, dtype=dtype)

        # initial condition
        rnn.set_state(init_state)

        # make prediction
        y = y0.detach()
        y_col = []
        z_col = []
        for step in range(steps):
            rnn.forward(inp[step])
            y = W_r @ rnn.z
            y_col.append(y)
            z_col.append(rnn.z)

        # save predictions
        predictions.append(np.asarray(y_col))
        model_dynamics.append(np.asarray(z_col))

# plotting
##########

# prediction figure
n_examples = 3
fig, axes = plt.subplots(ncols=2, nrows=n_examples, figsize=(12, 3*n_examples))
for i in range(n_examples):
    for j in range(2):
        ax = axes[i, j]
        ax.plot(targets[train_trials + i][:plot_steps, j], label="target", linestyle="dashed")
        ax.plot(predictions[i][:plot_steps, j], label="prediction", linestyle="solid")
        ax.set_ylabel(state_vars[j])
        ax.set_title(f"test trial = {i + 1}")
        if i == n_in-1:
            ax.set_xlabel("steps")
            ax.legend()
fig.suptitle("Model Predictions")
plt.tight_layout()

# model dynamics
fig, axes = plt.subplots(ncols=2, nrows=n_examples, figsize=(12, 3*n_examples))
n_neurons = 5
for i in range(n_examples):
    v = model_dynamics[i]
    ax = axes[i, 0]
    ax.plot(np.mean(v, axis=-1))
    ax.set_ylabel("v")
    ax.set_xlabel("time")
    ax.set_title("average model dynamics")
    ax = axes[i, 1]
    for j in range(n_neurons):
        ax.plot(v[:, j], label=f"neuron {j+1}")
    ax.set_ylabel("v")
    ax.set_xlabel("time")
    ax.set_title(f"single neuron dynamics")
    ax.legend()
fig.suptitle("Model dynamics")
plt.tight_layout()


plt.show()
