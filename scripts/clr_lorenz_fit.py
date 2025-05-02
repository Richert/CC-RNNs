""" Script for fitting a dRNN architecture with optional conceptor updates on the Lorenz DS reconstruction task.

Instructions:
- you don't need to run any input generation scripts. This is self-contained.
- by default, the following parameters are optimized (see line 102): L (line 90), W_in (line 91), W_r (line 83)
- a conceptor is trained for each condition. Conditions differ in their Lorenz equation parameters (line 44)
- important parameters to adjust for optimizing the training procedure are in lines 60-74. alpha and lam are the
  aperture and learning rate for conceptors, respectively. Everything else is backpropagation things.

"""
import sys
sys.path.append("../")
from src.rnn import LowRankCRNN
from src.functions import init_weights, init_dendrites
import torch
import numpy as np
import matplotlib.pyplot as plt

def lorenz(y: torch.Tensor, s: float = 10.0, r: float = 28.0, b: float = 2.667) -> torch.Tensor:
    y1, y2, y3 = y
    y1_dot = s*(y2 - y1)
    y2_dot = r*y1 - y2 - y1*y3
    y3_dot = y1*y2 - b*y3
    return torch.stack([y1_dot, y2_dot, y3_dot], dim=0)

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
visualize_results = True
plot_examples = 5

# task parameters
integration_steps = 10000
init_steps = 20
auto_steps = 100
y_init = 10.0
dim = 3
dt = 0.01
sampling_rate = 20
steps = int(integration_steps/sampling_rate)
conditions = {1: {"s": 10.0, "r": 28.0, "b": 2.667}}
n_conditions = len(conditions)
d = 1

# rnn parameters
k = 100
n_dendrites = 10
n_in = dim
n_out = dim
density = 0.5
in_scale = 0.1
out_scale = 0.2
Delta = 0.1
sigma = 0.8
N = int(k * n_dendrites)

# training parameters
trials = 1000
train_trials = int(0.9 * trials)
test_trials = trials - train_trials
lr = 1e-2
betas = (0.9, 0.999)
batch_size = 20
gradient_cutoff = 1e4
truncation_steps = 200
epsilon = 0.01
alpha = 3.0
lam = 1e-5
batches = int(train_trials / batch_size)
noise_lvl = 0.0

# model training
################

# initialize rnn matrices
bias = torch.tensor(Delta * np.random.randn(k), device=device, dtype=dtype)
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
L = init_weights(N, k, density)
W, R = init_dendrites(k, n_dendrites)
W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

# model initialization
rnn = LowRankCRNN(torch.tensor(W*0.0, dtype=dtype, device=device),
                  torch.tensor(L*sigma, dtype=dtype, device=device),
                  torch.tensor(R, device=device, dtype=dtype),
                  W_in, bias, g="ReLU", alpha=alpha, lam=lam)
rnn.free_param("W_in")
rnn.free_param("L")

# conceptor initialization
for c in conditions:
    rnn.init_new_y_controller(init_value="random")
    rnn.store_y_controller(c)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
rnn.clip(gradient_cutoff)

# training
train_loss = 0.0
loss_col = []
with torch.enable_grad():
    for batch in range(batches):

        loss = torch.zeros((1,), device=device, dtype=dtype)

        for trial in np.random.choice(train_trials, size=(batch_size,), replace=False):

            # choose condition randomly
            c = np.random.choice(conditions)

            # get initial state
            rnn.activate_y_controller(c)
            for step in range(init_steps):
                x = torch.randn(n_in, dtype=dtype, device=device)
                rnn.forward(x)
                rnn.update_y_controller()
            rnn.detach()

            # get input and target timeseries
            y = y_init * torch.randn(dim)
            y_col = []
            for step in range(integration_steps):
                y = y + dt * lorenz(y)
                if step % sampling_rate == 0:
                    y_col.append(y)
            y_col = torch.stack(y_col, dim=0)
            inp = y_col[:-d]
            target = y_col[d:]

            # collect loss
            y_col = []
            for step in range(inp.shape[0]):
                z = rnn.forward(inp[step])
                rnn.update_y_controller()
                y = W_r @ z
                if step % truncation_steps == truncation_steps - 1:
                    rnn.detach()
                y_col.append(y)

            # calculate loss
            y_col = torch.stack(y_col, dim=0)
            loss += loss_func(y_col, target)

            # store controller
            rnn.store_y_controller(conditions[trial])

        # make update
        optim.zero_grad()
        loss.backward()
        optim.step()
        rnn.detach()

        # store and print loss
        train_loss = loss.item()
        loss_col.append(train_loss)
        print(f"Training epoch {batch+1} / {batches}: MSE = {loss_col[-1]}")
        if train_loss < epsilon:
            break

# generate predictions
test_loss, predictions, targets, dynamics, test_conditions = [], [], [], [], []
with torch.no_grad():
    for trial in range(train_trials, trials):

        # choose condition randomly
        c = np.random.choice(conditions)

        # get initial state
        rnn.activate_y_controller(c)
        for step in range(init_steps):
            x = torch.randn(n_in, dtype=dtype, device=device)
            rnn.forward(x)

        # get input and target timeseries
        y = y_init * torch.randn(dim)
        y_col = []
        for step in range(integration_steps):
            y = y + dt * lorenz(y)
            if step % sampling_rate == 0:
                y_col.append(y)
        y_col = torch.stack(y_col, dim=0)
        inp = y_col[:-d]
        target = y_col[d:]

        # make prediction
        y_col, z_col = [], []
        for step in range(steps):
            x = inp[step] if step <= auto_steps else y
            z = rnn.forward(x)
            y = W_r @ z
            y_col.append(y)
            z_col.append(z)
        predictions.append(np.asarray(y_col))
        dynamics.append(np.asarray(z_col))

        # calculate loss
        loss = loss_func(torch.stack(y_col, dim=0), target)
        test_loss.append(loss.item())

if visualize_results:

    # prediction figure
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
    for i, trial in enumerate(np.random.choice(test_trials, size=(plot_examples,))):
        ax = axes[i]
        ax.plot(targets[train_trials + trial], label="targets", linestyle="dashed")
        for j, line in enumerate(ax.get_lines()):
            ax.plot(predictions[trial][:, j], label="predictions", linestyle="solid", color=line.get_color())
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
        mean_v = np.mean(dynamics[i], axis=-1)
        ax = axes[i]
        ax.plot(mean_v, color="black", label="mean")
        for j in np.random.choice(k, size=(n_neurons,)):
            ax.plot(dynamics[i][:, j], label=f"neuron {j + 1}")
        ax.axvline(x=int(0.5 * steps), color="grey", linestyle="dashed")
        ax.set_ylabel("amplitude")
        if i == plot_examples - 1:
            ax.set_xlabel("steps")
            ax.legend()
    fig.suptitle("RNN dynamics")
    plt.tight_layout()

    # conceptors figure
    conceptors = np.asarray([c.detach().cpu().numpy() for c in rnn.y_controllers.values()])
    fig, ax = plt.subplots(figsize=(12, 2 * len(conceptors)))
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

    plt.show()
