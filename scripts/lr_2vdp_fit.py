from src.rnn import LowRankRNN
from src.functions import init_weights, init_dendrites
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# function definitions
######################

def vanderpol(y1: np.ndarray, y2: np.ndarray, x: float, tau: float = 1.0) -> tuple:
    y1_dot = y2 / tau
    y2_dot = (y2*x*(1 - y1**2) - y1) / tau
    return y1_dot, y2_dot

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_examples = 5
state_vars = ["y1", "y2"]
visualization = {"connectivity": False, "inputs": True, "results": True}

# task parameters
mu = 0.5
min_tau = 0.2
d = 1
dt = 0.01
steps = 2000
init_steps = 1000

# rnn parameters
n_in = len(state_vars) + 1
n_out = len(state_vars)
k = 20
n_dendrites = 10
N = int(k*n_dendrites)
sr = 0.99
bias_scale = 0.01
bias = 0.0
in_scale = 1.0
out_scale = 0.02
density = 0.5

# training parameters
trials = 5000
train_trials = int(0.9*trials)
test_trials = trials - train_trials
lr = 5e-3
betas = (0.9, 0.999)
batch_size = 50
gradient_cutoff = 0.5/lr
truncation_steps = 1000
epsilon = 0.1

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias + bias_scale * np.random.randn(N), device=device, dtype=dtype)
L = np.abs(init_weights(N, k, density))
R = init_dendrites(k, n_dendrites, normalize=True)
W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

# plot connectivity
if visualization["connectivity"]:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    ax = axes[0]
    ax.imshow(L, aspect="auto", interpolation="none")
    ax.set_xlabel("from: neurons")
    ax.set_ylabel("to: dendrites")
    ax = axes[1]
    ax.imshow(R, aspect="auto", interpolation="none")
    ax.set_xlabel("from: dendrites")
    ax.set_ylabel("to: neurons")
    plt.tight_layout()
    plt.show()

# generate input and targets
############################

# generate targets and inputs
y0 = 2.0
targets, inputs, conditions = [], [], []
with torch.no_grad():
    for n in range(trials):
        successful = False
        tau = (1.0 - min_tau) * np.random.rand() + min_tau
        in_phase = np.random.randn() > 0
        while not successful:
            y1 = torch.zeros((2,), device=device, dtype=dtype) - y0
            y2 = torch.zeros((2,), device=device, dtype=dtype) + y0
            if not in_phase:
                y1[0], y2[0] = y0, -y0
            y_col = []
            for step in range(steps + d):
                y1_dot, y2_dot = vanderpol(y1, y2, x=mu, tau=tau)
                y1 = y1 + dt * y1_dot
                y2 = y2 + dt * y2_dot
                y_col.append(y1)
            y_col = np.asarray(y_col)
            if np.isfinite(y_col[-1, 0]):
                successful = True
        inp = np.zeros((steps + d, n_in))
        inp[:, :2] = y_col
        inp[:, 2] = tau
        inputs.append(inp)
        targets.append(y_col[d:])
        conditions.append(in_phase)

# plot targets
if visualization["inputs"]:
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
    for i in range(plot_examples):
        ax = axes[i]
        ax.plot(inputs[i][:, 0], label=state_vars[0])
        ax.plot(inputs[i][:, 1], label=state_vars[1])
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.set_title(f"training trial {i+1}: tau = {inputs[i][-1, 2]}, in-phase = {conditions[i]}")
        ax.legend()
    fig.suptitle("Target waveforms")
    plt.tight_layout()
    plt.show()

# train LR-RNN weights
######################

# initialize RFC
rnn = LowRankRNN(torch.tensor(L, dtype=dtype, device=device), torch.tensor(R, device=device, dtype=dtype), W_in, bias,
                 g="ReLU")
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
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
rnn.clip(gradient_cutoff)

# training
loss = torch.zeros((1,))
current_loss = 0.0
with torch.enable_grad():
    for trial in range(train_trials):

        # get input and target timeseries
        input = inputs[trial]
        target = torch.tensor(targets[trial], device=device, dtype=dtype)

        # initial condition
        rnn.detach()
        rnn.set_state(init_state)

        # collect loss
        y_col = []
        for step in range(steps):
            inp = torch.tensor(input[step], device=device, dtype=dtype)
            if step > init_steps:
                inp = torch.concat((y, inp[n_out:]), dim=0)
            rnn.forward(inp)
            y = W_r @ rnn.z
            y_col.append(y)
            if step > init_steps:
                inp[step+d, :n_out] = y.detach().cpu().numpy()

        # calculate loss
        y_col = torch.stack(y_col, dim=0)
        loss += loss_func(y_col, target)

        # make update
        if trial % batch_size == batch_size - 1:
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % truncation_steps == truncation_steps - 1:
                rnn.detach()
            current_loss = loss.item()
            loss = torch.zeros((1,))
            print(f"Training trial {trial + 1} / {train_trials}: MSE = {current_loss}")
            if current_loss < epsilon:
                break

# generate predictions
######################

predictions = []
z_dynamics = []
with torch.no_grad():
    for trial in range(train_trials, trials):

        # get input and target timeseries
        input = inputs[trial]

        # initial condition
        rnn.set_state(init_state)

        # make prediction
        y_col = []
        z_col = []
        for step in range(steps):
            rnn.forward(torch.tensor(inp[step], device=device, dtype=dtype))
            y = W_r @ rnn.z
            y_col.append(y)
            z_col.append(rnn.z)
            inp[step+d, :n_out] = y.detach().cpu().numpy()

        # save predictions
        predictions.append(np.asarray(y_col))
        z_dynamics.append(np.asarray(z_col))

# plotting
##########

if visualization["results"]:

    # prediction figure
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
    for i in range(plot_examples):
            ax = axes[i]
            ax.plot(targets[train_trials + i][:, 0], label="target 1", linestyle="dashed", color="black")
            ax.plot(targets[train_trials +i][:, 1], label="target 2", linestyle="dashed", color="darkorange")
            ax.plot(predictions[i][:, 0], label="prediction 1", linestyle="solid", color="black")
            ax.plot(predictions[i][:, 1], label="prediction 2", linestyle="solid", color="darkorange")
            ax.set_ylabel("amplitude")
            ax.set_title(f"test trial {i + 1}")
            if i == plot_examples-1:
                ax.set_xlabel("steps")
                ax.legend()
    fig.suptitle("Model Predictions")
    plt.tight_layout()

    # dynamics figure
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
    n_neurons = 5
    for i in range(plot_examples):
        mean_v = np.mean(z_dynamics[i], axis=-1)
        ax = axes[i]
        ax.plot(mean_v, color="black", label="mean")
        for j in range(n_neurons):
            ax.plot(z_dynamics[i][:, j], label=f"neuron {j+1}")
        ax.set_ylabel("amplitude")
        if i == plot_examples - 1:
            ax.set_xlabel("steps")
            ax.legend()
    fig.suptitle("RNN dynamics")
    plt.tight_layout()

    plt.show()
