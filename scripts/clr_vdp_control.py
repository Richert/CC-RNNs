from src.rnn import LowRankCRNN
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
plot_examples = 6
state_vars = ["y1", "y2"]
visualization = {"connectivity": True, "inputs": True, "results": True}
n_conditions = 3

# load inputs and targets
data = pickle.load(open(f"../data/{n_conditions}vdp_inputs_sr20.pkl", "rb"))
targets = data["targets"]
inputs = data["inputs"]
conditions = data["trial_conditions"]
unique_conditions = np.unique(conditions)

# task parameters
steps = inputs[0].shape[0]
init_steps = 1000
switch_test_cond = True
noise_lvl = 0.5

# finalize inputs
inputs = []
for inp, cond in zip(data["inputs"], conditions):
    new_inp = np.zeros((inp.shape[0], 1))
    new_inp[:, 0] = inp + noise_lvl * np.random.randn(inp.shape[0])
    inputs.append(new_inp)

# rnn parameters
n_in = inputs[0].shape[-1] if len(inputs[0].shape) > 1 else 1
n_out = targets[0].shape[-1]
k = 20
n_dendrites = 10
N = int(k * n_dendrites)
sr = 0.99
bias_scale = 0.01
bias = 0.0
in_scale = 1.0
out_scale = 0.02
density = 0.5
min_dendrite = 0.1

# training parameters
trials = len(conditions)
train_trials = int(0.9 * trials)
test_trials = trials - train_trials
augmentation = 2.0
lr = 5e-3
betas = (0.9, 0.999)
batch_size = 50
gradient_cutoff = 0.5 / lr
truncation_steps = 1000
epsilon = 0.1
alpha = 20.0
lam = 1e-3
batches = int(augmentation * train_trials / batch_size)

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias + bias_scale * np.random.randn(N), device=device, dtype=dtype)
L = np.abs(init_weights(N, k, density))
R = init_dendrites(k, n_dendrites, normalize=True, min_dendrite=0.1)
W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

# plot connectivity
if visualization["connectivity"]:
    fig_conn, axes_conn = plt.subplots(ncols=2, nrows=2, figsize=(12, 5))
    ax = axes_conn[0, 0]
    ax.imshow(L, aspect="auto", interpolation="none")
    ax.set_xlabel("from: neurons")
    ax.set_ylabel("to: dendrites")
    ax.set_title("Pre-training synaptic weights")
    ax = axes_conn[0, 1]
    ax.imshow(R, aspect="auto", interpolation="none")
    ax.set_title("Pre-training dendritic tree")
    ax.set_xlabel("from: dendrites")
    ax.set_ylabel("to: neurons")

# plot inputs and targets
if visualization["inputs"]:
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
    for i, trial in enumerate(np.random.choice(train_trials, size=(plot_examples,))):
        ax = axes[i]
        ax.plot(inputs[trial], label="x")
        ax.plot(targets[trial][:, 0], label=state_vars[0])
        ax.plot(targets[trial][:, 1], label=state_vars[1])
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.set_title(f"training trial {trial + 1}: in-phase = {conditions[trial]}")
        ax.legend()
    fig.suptitle("Inputs (x) and Target Waveforms (y)")
    plt.tight_layout()

# train LR-RNN weights
######################

# initialize RFC
rnn = LowRankCRNN(torch.tensor(L, dtype=dtype, device=device), torch.tensor(R, device=device, dtype=dtype),
                  W_in, bias, alpha=alpha, lam=lam, g="ReLU")
rnn.free_param("W")
rnn.free_param("W_in")
rnn.free_param("bias")

# initial wash-out period
avg_input = torch.zeros((n_in,), device=device, dtype=dtype)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)
init_state = (v[:] for v in rnn.state_vars)

# initialize conceptors
other_conditions = {}
for c in unique_conditions:
    rnn.init_new_y_controller(init_value="random")
    rnn.store_y_controller(c)
    other_conditions[c] = unique_conditions[unique_conditions != c]

# choose conceptor to be trained in left-out condition
loc = unique_conditions[-1]
rnn.y_controllers[loc].requires_grad = True

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizers
optim1 = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
optim2 = torch.optim.Adam([rnn.y_controllers[loc], W_r], lr=lr, betas=betas)
rnn.clip(gradient_cutoff)
rnn.y_controllers[loc].register_hook(lambda grad: torch.clamp(grad, -gradient_cutoff, gradient_cutoff))
optimizers = {False: optim1, True: optim2}

# set up training trials
indices = {False: np.asarray(conditions)[:train_trials] != loc,
           True: np.asarray(conditions)[:train_trials] == loc}
trial_indices = np.arange(train_trials)

# training
current_loss = 0.0
loss_col = []
with torch.enable_grad():
    for batch in range(batches):

        loss = torch.zeros((1,))
        batch_condition = np.random.choice(unique_conditions)
        optim = optimizers[batch_condition == loc]
        batch_trials = trial_indices[indices[batch_condition == loc]]

        for trial in np.random.choice(batch_trials, size=(batch_size,), replace=False):

            # get input and target timeseries
            inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
            target = torch.tensor(targets[trial], device=device, dtype=dtype)

            # initial condition
            rnn.detach()
            rnn.set_state(init_state)
            rnn.activate_y_controller(conditions[trial])

            # collect loss
            y_col = []
            for step in range(steps):
                z = rnn.forward(inp[step])
                rnn.update_y_controller()
                y = W_r @ z
                y_col.append(y)

            # calculate loss
            y_col = torch.stack(y_col, dim=0)
            loss += loss_func(y_col, target)

        # make update
        if batch < batches - 2:
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % truncation_steps == truncation_steps - 1:
                rnn.detach()
            current_loss = loss.item()
            loss_col.append(current_loss)
            print(f"Training batch {batch + 1} / {batches}: MSE = {current_loss}")
            if current_loss < epsilon:
                break

# generate predictions
######################

predictions = []
z_dynamics = []
switch_condition = []
with torch.no_grad():
    for trial in range(train_trials, trials):

        # get input and target timeseries
        inp = torch.tensor(inputs[trial], device=device, dtype=dtype)

        # initial condition
        rnn.set_state(init_state)
        rnn.activate_y_controller(conditions[trial])

        # switch condition
        c_switch = np.random.choice(other_conditions[conditions[trial]])

        # make prediction
        y_col = []
        z_col = []
        for step in range(steps):
            z = rnn.forward(inp[step])
            rnn.update_y_controller()
            y = W_r @ z
            y_col.append(y)
            z_col.append(z)
            if step == int(0.5 * steps):
                rnn.activate_y_controller(c_switch)

        # save predictions
        predictions.append(np.asarray(y_col))
        z_dynamics.append(np.asarray(z_col))
        switch_condition.append(c_switch)

# plotting
##########

if visualization["results"]:

    # prediction figure
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
    for i, trial in enumerate(np.random.choice(test_trials, size=(plot_examples,))):
        ax = axes[i]
        ax.plot(targets[train_trials + trial][:, 0], label="target 1", linestyle="dashed", color="black")
        ax.plot(targets[train_trials + trial][:, 1], label="target 2", linestyle="dashed", color="darkorange")
        ax.plot(predictions[trial][:, 0], label="prediction 1", linestyle="solid", color="black")
        ax.plot(predictions[trial][:, 1], label="prediction 2", linestyle="solid", color="darkorange")
        ax.axvline(x=int(0.5 * steps), color="grey", linestyle="dashed")
        ax.set_ylabel("amplitude")
        ax.set_title(f"test trial {trial + 1}: switch condition {switch_condition[trial]}")
        if i == plot_examples - 1:
            ax.set_xlabel("steps")
            ax.legend()
    fig.suptitle("Model Predictions")
    plt.tight_layout()

    # dynamics figure
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
    n_neurons = 5
    for i in range(plot_examples):
        mean_v = np.mean(z_dynamics[i], axis=-1)
        ax = axes[i]
        ax.plot(mean_v, color="black", label="mean")
        for j in np.random.choice(k, size=(n_neurons,)):
            ax.plot(z_dynamics[i][:, j], label=f"neuron {j + 1}")
        ax.axvline(x=int(0.5 * steps), color="grey", linestyle="dashed")
        ax.set_ylabel("amplitude")
        if i == plot_examples - 1:
            ax.set_xlabel("steps")
            ax.legend()
    fig.suptitle("RNN dynamics")
    plt.tight_layout()

    # conceptors figure
    conceptors = np.asarray([c.detach().cpu().numpy() for c in rnn.y_controllers.values()])
    fig, ax = plt.subplots(figsize=(12, 2*len(conceptors)))
    im = ax.imshow(conceptors, aspect="auto", interpolation="none", cmap="cividis")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("neurons")
    ax.set_ylabel("conditions")
    ax.set_title("Conceptors")
    plt.tight_layout()

    # training loss figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(loss_col)
    ax.set_xlabel("training batch")
    ax.set_ylabel("MSE")
    ax.set_title("Training loss")
    plt.tight_layout()

# plot connectivity
if visualization["connectivity"]:
    ax = axes_conn[1, 0]
    ax.imshow(rnn.W.detach().cpu().numpy(), aspect="auto", interpolation="none")
    ax.set_xlabel("from: neurons")
    ax.set_ylabel("to: dendrites")
    ax.set_title("Post-training synaptic weights (should change)")
    ax = axes_conn[1, 1]
    ax.imshow(rnn.W_z.detach().cpu().numpy(), aspect="auto", interpolation="none")
    ax.set_xlabel("from: dendrites")
    ax.set_ylabel("to: neurons")
    ax.set_title("Post-training dendritic tree (shouldn't change)")
    plt.sca(ax)
    plt.tight_layout()

    plt.show()
