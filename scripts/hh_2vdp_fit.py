from src.rnn import HHRNN
from src.functions import init_weights, init_dendrites
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_examples = 5
state_vars = ["y1", "y2"]
visualization = {"connectivity": False, "inputs": False, "results": False}

# load inputs and targets
path = "/home/richard/PycharmProjects/CC-RNNs"
data = pickle.load(open(f"{path}/data/vanderpol_inputs_sr1.pkl", "rb"))
inputs = data["inputs"]
targets = data["targets"]
conditions = data["trial_conditions"]

# task parameters
steps = inputs[0].shape[0]
init_steps = 1000
switch_test_cond = True

# HH parameters
theta = 40.0
gamma = 10.0
dt = 1e-2
t_scale = 10

# rnn parameters
n_in = inputs[0].shape[-1] if len(inputs[0].shape) > 1 else 1
n_out = targets[0].shape[-1]
k = 20
n_dendrites = 10
N = int(k*n_dendrites)
sr = 0.99
bias_scale = 1.0
bias = 10.0
in_scale = 1.0
out_scale = 0.02
density = 0.5

# training parameters
trials = len(conditions)
train_trials = int(0.9*trials)
test_trials = trials - train_trials
lr = 1e-3
betas = (0.9, 0.999)
batch_size = 20
gradient_cutoff = 0.5/lr
truncation_steps = 200
epsilon = 0.1
alpha = 20.0
lam = 1e-4

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

# plot inputs and targets
if visualization["inputs"]:
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
    for i, trial in enumerate(np.random.choice(train_trials, size=(plot_examples,))):
        ax = axes[i]
        ax.plot(inputs[trial], label="x")
        ax.plot(targets[trial][:, 0], label=state_vars[0])
        ax.plot(targets[trial][:, 1], label=state_vars[1])
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.set_title(f"training trial {trial+1}: in-phase = {conditions[trial]}")
        ax.legend()
    fig.suptitle("Inputs (x) and Target Waveforms (y)")
    plt.tight_layout()
    plt.show()

# train LR-RNN weights
######################

# initialize RFC
rnn = HHRNN(torch.tensor(L, dtype=dtype, device=device), torch.tensor(R, device=device, dtype=dtype),
                          W_in, bias, alpha=alpha, lam=lam, theta=theta, gamma=gamma, dt=dt, g="ReLU")
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

# initialize conceptors
for c in np.unique(conditions):
    rnn.init_new_conceptor(init_value="random")
    rnn.store_conceptor(c)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
rnn.clip(gradient_cutoff)

# training
loss = torch.zeros((1,))
current_loss = 0.0
with torch.enable_grad():
    for i, trial in enumerate(np.random.permutation(train_trials)):

        # get input and target timeseries
        inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
        target = torch.tensor(targets[trial], device=device, dtype=dtype)

        # initial condition
        rnn.detach()
        rnn.set_state(init_state)
        rnn.activate_conceptor(conditions[trial])

        # collect loss
        y_col = []
        for step in range(steps):
            out = []
            for _ in range(t_scale):
                rnn.forward_c_adapt(inp[step:step+1])
                out.append(rnn.f(rnn.z))
            y = W_r @ torch.mean(torch.stack(out, dim=0), dim=0)
            y_col.append(y)

        # calculate loss
        y_col = torch.stack(y_col, dim=0)
        loss += loss_func(y_col, target)

        # make update
        if i % batch_size == batch_size - 1 and i < train_trials - 2*batch_size:
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % truncation_steps == truncation_steps - 1:
                rnn.detach()
            current_loss = loss.item()
            loss = torch.zeros((1,))
            print(f"Training trial {i + 1} / {train_trials}: MSE = {current_loss}")
            if current_loss < epsilon:
                break

# generate predictions
######################

predictions = []
z_dynamics = []
with torch.no_grad():
    for trial in range(train_trials, trials):

        # get input and target timeseries
        inp = torch.tensor(inputs[trial], device=device, dtype=dtype)

        # initial condition
        rnn.set_state(init_state)
        rnn.activate_conceptor(conditions[trial])

        # make prediction
        y_col = []
        z_col = []
        for step in range(steps):
            out = []
            for _ in range(t_scale):
                rnn.forward_c(inp[step:step + 1])
                out.append(rnn.f(rnn.z))
            z = torch.mean(torch.stack(out, dim=0), dim=0)
            y = W_r @ z
            y_col.append(y)
            z_col.append(z)
            if step > 0.5*steps:
                rnn.activate_conceptor(False if conditions[trial] else True)

        # save predictions
        predictions.append(np.asarray(y_col))
        z_dynamics.append(np.asarray(z_col))

# plotting
##########

if visualization["results"]:

    # prediction figure
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
    for i, trial in enumerate(np.random.choice(test_trials, size=(plot_examples,))):
            ax = axes[i]
            ax.plot(targets[train_trials + trial][:, 0], label="target 1", linestyle="dashed", color="black")
            ax.plot(targets[train_trials +trial][:, 1], label="target 2", linestyle="dashed", color="darkorange")
            ax.plot(predictions[trial][:, 0], label="prediction 1", linestyle="solid", color="black")
            ax.plot(predictions[trial][:, 1], label="prediction 2", linestyle="solid", color="darkorange")
            ax.set_ylabel("amplitude")
            ax.set_title(f"test trial {trial + 1}")
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
        for j in np.random.choice(k, size=(n_neurons,)):
            ax.plot(z_dynamics[i][:, j], label=f"neuron {j+1}")
        ax.set_ylabel("amplitude")
        if i == plot_examples - 1:
            ax.set_xlabel("steps")
            ax.legend()
    fig.suptitle("RNN dynamics")
    plt.tight_layout()

    # conceptors figure
    fig, ax = plt.subplots(figsize=(12, 4))
    conceptors = np.asarray([c for c in rnn.conceptors.values()])
    im = ax.imshow(conceptors, aspect="auto", interpolation="none", cmap="cividis")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("neurons")
    ax.set_ylabel("conditions")
    ax.set_title("Conceptors")
    plt.tight_layout()

    plt.show()
