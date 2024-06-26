from src import LowRankRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights


# function definitions
######################

def get_inp(f: float, trial_dur: int, trials: int, noise: float, dt: float, single_output: bool = True
            ) -> tuple:

    # create sines
    time = np.linspace(0.0, trial_dur*dt, trial_dur)
    s = np.sin(2.0*np.pi*freq*time)

    # create switching signal and targets
    inputs = np.zeros((trials, trial_dur, 1))
    targets = np.zeros((trials, trial_dur, 1 if single_output else 2))
    for trial in range(trials):
        if np.random.randn() > 0:
            inputs[trial, :, 0] = 1.0
            targets[trial, :, 0 if single_output else 1] = s

    # add noise to input signals
    inputs += noise * np.random.randn(trials, trial_dur, 1)

    return torch.tensor(inputs, device=device, dtype=dtype), torch.tensor(targets, device=device, dtype=dtype)


def cc_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, padding: int = 1) -> torch.Tensor:
    if torch.max(y) > 0.5:
        cc = torch.abs(torch.nn.functional.conv1d(x[None, None, :], y[None, None, :], padding=padding))
        return -torch.max(cc) + alpha*(torch.abs(torch.max(x) - 1.0) + torch.abs(torch.min(x) + 1.0))
    else:
        return alpha*torch.mean(x)


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 1000

# input parameters
freq = 10.0
dt = 0.01
n_in = 1
trial_dur = 100
padding = int(0.4*trial_dur)
noise_lvl = 0.01

# reservoir parameters
N = 200
n_out = 1
k = 5
sr = 1.05
bias_scale = 0.01
in_scale = 0.1
density = 0.2
out_scale = 0.1
init_noise = 0.05

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
n_train = 500000
n_test = 1000
init_steps = 1000
batch_size = 20
lr = 0.0005
betas = (0.9, 0.999)
alphas = (5.0, 1e-4)

# generate inputs and targets
#############################

# get training data
x_train, y_train = get_inp(freq, trial_dur, n_train, noise_lvl, dt)

# get test data
x_test, y_test = get_inp(freq, trial_dur, n_test, noise_lvl, dt)

# training
##########

# initialize LR-RNN
rnn = LowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias, torch.tensor(W_z, device=device, dtype=dtype))
rnn.free_param("W")
rnn.free_param("W_z")

# set up loss function
loss_func = cc_loss

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# initial wash-out period
y_col = []
for step in range(init_steps):
    y_col.append(rnn.forward(init_noise*torch.randn(n_in, dtype=dtype, device=device)).cpu().detach().numpy())
fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(np.asarray(y_col).T, aspect="auto", interpolation="none", cmap="viridis", vmin=-1.0, vmax=1.0)
plt.colorbar(im, ax=ax, shrink=0.6)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
plt.tight_layout()
plt.show()

# training
current_loss = 100.0
min_loss = -50.0
loss_hist = []
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train):

        # trial
        trial_inp = x_train[trial]
        trial_targ = y_train[trial]
        outputs = []
        for step in range(trial_dur):
            outputs.append(W_r @ rnn.forward(trial_inp[step]))
        outputs = torch.stack(outputs, dim=0)
        for i in range(n_out):
            loss += loss_func(outputs[:, i], trial_targ[:, i], alpha=alphas[0], padding=padding)

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size*trial_dur
            loss += alphas[1] * torch.sum(torch.abs(rnn.W) @ torch.abs(rnn.W_z))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()
            loss_hist.append(current_loss)
            print(f"MSE loss after {trial+1} training trials: {current_loss}")

        if current_loss < min_loss:
            break

W = (rnn.W @ rnn.W_z).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(rnn.W_z)).cpu().detach().numpy())

# testing
#########

# generate predictions
predictions, targets = [], []
with torch.no_grad():
    for trial in range(n_test):

        trial_inp = x_test[trial]
        trial_targ = y_test[trial]

        for step in range(trial_dur):

            y = W_r @ rnn.forward(trial_inp[step])
            predictions.append(y.cpu().detach().numpy())
            targets.append(trial_targ[step])

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)
performance = np.mean((predictions.squeeze() - targets.squeeze())**2)

# plotting
##########

# dynamics
_, axes = plt.subplots(nrows=n_out, figsize=(12, 3*n_out), sharex=True)
for i in range(n_out):
    ax = axes[i] if n_out > 1 else axes
    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(f"y_{i+1}")
    if i == n_out-1:
        ax.set_xlabel("steps")
        ax.legend()
    if i == 0:
        ax.set_title(f"Test error: {np.round(performance, decimals=2)}")
plt.tight_layout()

# trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Absolute weights: {np.round(W_abs, decimals=1)}")

# loss history
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(loss_hist)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.tight_layout()
plt.show()
