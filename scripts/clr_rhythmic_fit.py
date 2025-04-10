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
state_vars = ["y"]
visualization = {"connectivity": True, "inputs": True, "results": True}

# load inputs and targets
data = pickle.load(open("../data/cosine_inputs_3f.pkl", "rb"))
inputs = data["inputs"]
targets = data["targets"]
conditions = data["trial_conditions"]

# task parameters
steps = inputs[0].shape[0]
init_steps = 1000
noise_lvl = 0.01

# add noise to input
inputs = [inp + noise_lvl * np.random.randn(*inp.shape) for inp in inputs]

# rnn parameters
n_in = inputs[0].shape[-1] if len(inputs[0].shape) > 1 else 1
n_out = 1
k = 20
n_dendrites = 20
N = int(k * n_dendrites)
sr = 0.99
bias_scale = 0.01
bias = 0.0
in_scale = 1.0
out_scale = 0.02
density = 0.5
min_dendrite = 0.1
k_dendrite = 0.2

# training parameters
trials = len(conditions)
train_trials = int(0.9 * trials)
test_trials = trials - train_trials
augmentation = 1.0
lr = 5e-3
betas = (0.9, 0.999)
batch_size = 50
gradient_cutoff = 0.5 / lr
truncation_steps = 100
epsilon = 1.0
alpha = 30.0
lam = 5e-3
batches = int(augmentation * train_trials / batch_size)

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias + bias_scale * np.random.randn(k), device=device, dtype=dtype)
L = init_weights(N, k, density)
W, R = init_dendrites(k, n_dendrites, normalize=True, min_dendrite=min_dendrite)
W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

# plot connectivity
if visualization["connectivity"]:
    fig_conn, axes_conn = plt.subplots(ncols=3, nrows=2, figsize=(12, 5))
    ax = axes_conn[0, 0]
    ax.imshow(L, aspect="auto", interpolation="none")
    ax.set_xlabel("from: neurons")
    ax.set_ylabel("to: dendrites")
    ax.set_title("Synaptic weights (should change)")
    ax = axes_conn[0, 1]
    ax.imshow(R, aspect="auto", interpolation="none")
    ax.set_title("Dendrite-soma interactions (shouldn't change)")
    ax.set_xlabel("from: dendrites")
    ax.set_ylabel("to: neurons")
    ax = axes_conn[0, 2]
    ax.imshow(W, aspect="auto", interpolation="none")
    ax.set_title("Dendritic interactions (shouldn't change)")
    ax.set_xlabel("from: dendrites")
    ax.set_ylabel("to: dendrites")
    plt.tight_layout()

# plot inputs and targets
if visualization["inputs"]:
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
    for i, trial in enumerate(np.random.choice(train_trials, size=(plot_examples,))):
        ax = axes[i]
        ax.plot(inputs[trial], label="x")
        ax.plot(targets[trial], label=state_vars[0])
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.set_title(f"training trial {trial + 1}: in-phase = {conditions[trial]}")
        ax.legend()
    fig.suptitle("Inputs (x) and Target Waveforms (y)")
    plt.tight_layout()

# train LR-RNN weights
######################

# initialize RFC
rnn = LowRankCRNN(torch.tensor(W*k_dendrite, dtype=dtype, device=device),
                  torch.tensor(L, dtype=dtype, device=device),
                  torch.tensor(R, device=device, dtype=dtype),
                  W_in, bias, alpha=alpha, lam=lam, g="ReLU")
rnn.free_param("W_in")
rnn.free_param("bias")
rnn.free_param("L")

# initial wash-out period
avg_input = torch.zeros((n_in,), device=device, dtype=dtype)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)
init_state = (v[:] for v in rnn.state_vars)

# initialize z controllers
unique_conditions = np.unique(conditions)
other_conditions = {}
for c in unique_conditions:
    rnn.init_new_y_controller(init_value="random")
    rnn.store_y_controller(c)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
rnn.clip(gradient_cutoff)

# training
current_loss = 0.0
loss_col = []
with torch.enable_grad():
    for batch in range(batches):

        loss = torch.zeros((1,))

        for trial in np.random.choice(train_trials, size=(batch_size,), replace=False):

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
                z = rnn.forward(inp[step:step + 1])
                rnn.update_y_controller()
                y = W_r @ z
                y_col.append(y)

            # calculate loss
            y_col = torch.stack(y_col, dim=0).squeeze()
            loss += loss_func(y_col, target)

        # make update
        if batch < batches - 10:
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % truncation_steps == truncation_steps - 1:
                rnn.detach()

        # store and print loss
        current_loss = loss.item()
        loss_col.append(current_loss)
        print(f"Training batch {batch + 1} / {batches}: MSE = {current_loss}")
        if current_loss < epsilon:
            break

# generate predictions
######################

predictions = []
z_dynamics = []
test_loss = []
with torch.no_grad():
    for trial in range(train_trials, trials):

        # get input and target timeseries
        inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
        target = torch.tensor(targets[trial], device=device, dtype=dtype)

        # initial condition
        rnn.set_state(init_state)
        rnn.activate_y_controller(conditions[trial])

        # make prediction
        y_col = []
        z_col = []
        for step in range(steps):
            z = rnn.forward(inp[step:step + 1])
            y = W_r @ z
            y_col.append(y)
            z_col.append(z)

        # calculate loss
        loss = loss_func(torch.stack(y_col, dim=0).squeeze(), target)

        # save predictions
        predictions.append(np.asarray(y_col))
        z_dynamics.append(np.asarray(z_col))
        test_loss.append(loss.item())

# save results
results = {"predictions": predictions, "z_dynamics": z_dynamics, "train_loss": loss_col, "test_loss": test_loss,
           "W_r": W_r, "y_controllers": {key: c.detach().cpu().numpy() for key, c in rnn.y_controllers.items()},
           }
results.update({key: getattr(rnn, key).detach().cpu().numpy() for key in ["L", "R", "W", "W_in", "bias"]})
pickle.dump(results, open(f"../data/clr_rhythmic_fit.pkl", "wb"))

# plotting
##########

if visualization["results"]:

    # prediction figure
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
    for i, trial in enumerate(np.random.choice(test_trials, size=(plot_examples,))):
        ax = axes[i]
        ax.plot(targets[train_trials + trial], label="target", linestyle="dashed", color="black")
        ax.plot(predictions[trial], label="prediction", linestyle="solid", color="darkorange")
        ax.axvline(x=int(0.5 * steps), color="grey", linestyle="dashed")
        ax.set_ylabel("amplitude")
        ax.set_title(f"test trial {trial + 1}")
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
    ax.imshow(rnn.L.detach().cpu().numpy(), aspect="auto", interpolation="none")
    ax.set_xlabel("from: neurons")
    ax.set_ylabel("to: dendrites")
    ax = axes_conn[1, 1]
    ax.imshow(rnn.R.detach().cpu().numpy(), aspect="auto", interpolation="none")
    ax.set_xlabel("from: dendrites")
    ax.set_ylabel("to: neurons")
    ax = axes_conn[1, 2]
    ax.imshow(rnn.W.detach().cpu().numpy(), aspect="auto", interpolation="none")
    ax.set_xlabel("from: dendrites")
    ax.set_ylabel("to: dendrites")
    plt.tight_layout()

    plt.show()
