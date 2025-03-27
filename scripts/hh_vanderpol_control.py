from src.rnn import HHRNN
from src.functions import init_weights, init_dendrites
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# function definitions
######################

def vanderpol(y: torch.Tensor, x: torch.Tensor, mu: float, alpha: float = 0.1) -> torch.Tensor:
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
noise_lvl = 1.0
sigma = 500
dt = 1e-2
steps = 20000
init_steps = 1000
truncation_steps = 2000
gradient_cutoff = 1e3

# HH parameters
theta = 40.0
gamma = 10.0

# rnn parameters
n_in = 3
n_out = 1
k = 20
n_dendrites = 10
N = int(k*n_dendrites)
sr = 0.99
bias_scale = 0.01
in_scale = 1.0
out_scale = 0.02
density = 0.5

# training parameters
trials = 100
train_trials = int(0.9*trials)
test_trials = trials - train_trials
lr = 1e-2
betas = (0.9, 0.999)
batch_size = 50

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
L = np.abs(init_weights(N, k, density))
R = init_dendrites(k, n_dendrites, normalize=True)
W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

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

# generate targets
y0 = torch.zeros((2,), device=device, dtype=dtype)
y0[0] = 1e-1
targets = {}
with torch.no_grad():
    for i in range(n_in):
        mu = (max_mu - min_mu) * np.random.rand() + min_mu
        y = y0.detach()
        x = gaussian_filter1d(np.random.randn(steps), sigma=sigma)
        x *= noise_lvl / np.max(x)
        x = torch.tensor(x, dtype=dtype, device=device)
        y_col = []
        for step in range(steps):
            y = y + dt * vanderpol(y, x=x[step], mu=mu)
            y_col.append(y.detach().cpu().numpy())
        y_col = np.asarray(y_col)
        targets[i] = torch.tensor(y_col / np.max(y_col), dtype=dtype, device=device)

# generate inputs
inputs = [np.random.choice(n_in) for _ in range(trials)]

# plot targets
fig, axes = plt.subplots(nrows=n_in, figsize=(12, 3*n_in))
for i in range(n_in):
    ax = axes[i]
    ax.plot(targets[i][:, 0], label="x")
    ax.plot(targets[i][:, 1], label="y")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.legend()
fig.suptitle("Target waveforms")
plt.tight_layout()
plt.show()

# train LR-RNN weights
######################

# initialize RFC
rnn = HHRNN(torch.tensor(L, dtype=dtype, device=device), torch.tensor(R, device=device, dtype=dtype), W_in, bias,
            theta=theta, gamma=gamma, dt=dt)
rnn.free_param("W")
rnn.free_param("W_z")
rnn.free_param("W_in")
rnn.free_param("bias")

# initial wash-out period
avg_input = torch.zeros((n_in,), device=device, dtype=dtype)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)
init_state = rnn.y[:]

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
rnn.clip(gradient_cutoff)

# training
loss = torch.zeros((1,))
current_loss = 0.0
with torch.enable_grad():
    for trial in range(train_trials):

        # define input timeseries
        cond = inputs[trial]
        inp = torch.zeros((steps, n_in), device=device, dtype=dtype)
        inp[:, cond] = 1.0

        # initial condition
        rnn.detach()
        rnn.set_state(init_state)

        # collect loss
        y = y0.detach()
        y_col = []
        for step in range(steps):
            rnn.forward(inp[step])
            x = W_r @ rnn.f(rnn.z)
            y = y + dt * vanderpol(y, x=x[0], mu=mu)
            if step % truncation_steps == truncation_steps - 1:
                rnn.detach()
            y_col.append(y)

        # calculate loss
        y_col = torch.stack(y_col, dim=0)
        y_col = y_col / torch.max(y_col)
        loss += loss_func(y_col, targets[cond])

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

predictions = {key: [] for key in targets.keys()}
rnn_dynamics = {key: [] for key in targets.keys()}
loss = torch.zeros((1,))
with torch.no_grad():
    for trial in range(train_trials, trials):

        # define input timeseries
        cond = inputs[trial]
        inp = torch.zeros((steps, n_in), device=device, dtype=dtype)
        inp[:, cond] = 1.0

        # initial condition
        rnn.set_state(init_state)

        # training the RNN weights
        current_loss = 0.0
        y = y0.detach()
        y_col = []
        z_col = []
        for step in range(steps):
            rnn.forward(inp[step])
            x = W_r @ rnn.f(rnn.z)
            y = y + dt * vanderpol(y, x=x[0], mu=mu)
            y_col.append(y.detach().cpu().numpy())
            z_col.append(rnn.z.detach().cpu().numpy())

        # save predictions
        y_col = np.asarray(y_col)
        z_col = np.asarray(z_col)
        predictions[cond].append(y_col / np.max(y_col))
        rnn_dynamics[cond].append(z_col)

# plotting
##########

# prediction figure
fig, axes = plt.subplots(ncols=2, nrows=n_in, figsize=(12, 9))
for i in range(n_in):
    for j in range(2):
        ax = axes[i, j]
        l = ax.plot(targets[i][:plot_steps, j], label="target", linestyle="dashed")
        mean_prediction = np.mean(predictions[i], axis=0)[:plot_steps, j]
        ax.plot(mean_prediction, label="prediction", linestyle="solid")
        ax.set_ylabel(state_vars[j])
        ax.set_title(f"condition = {i + 1}")
        if i == n_in-1:
            ax.set_xlabel("steps")
            ax.legend()
fig.suptitle("Model Predictions")
plt.tight_layout()

# dynamics figure
fig, axes = plt.subplots(ncols=2, nrows=n_in, figsize=(12, 3*n_in))
n_neurons = 5
for i in range(n_in):
    mean_v = np.mean(rnn_dynamics[i], axis=0)
    ax = axes[i, 0]
    ax.plot(np.mean(mean_v, axis=-1))
    ax.set_ylabel("v")
    ax.set_xlabel("time")
    ax.set_title("average model dynamics")
    ax = axes[i, 1]
    for j in range(n_neurons):
        ax.plot(mean_v[:, j], label=f"neuron {j+1}")
    ax.set_ylabel("v")
    ax.set_xlabel("time")
    ax.set_title(f"Trial-averaged neuron dynamics")
    ax.legend()
fig.suptitle("Model dynamics")
plt.tight_layout()

plt.show()
