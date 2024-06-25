from src import LowRankRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights


# function definitions
######################

def get_inp(f1: float, f2: float, trial_dur: int, trials: int, noise: float, dt: float) -> tuple:

    # create sines
    time = np.linspace(0.0, trial_dur*dt, trial_dur)
    s1 = np.sin(2.0*np.pi*f1*time)
    s2 = np.sin(2.0*np.pi*f2*time)

    # create switching signal and targets
    inputs = np.zeros((trials, trial_dur, 1))
    targets = np.zeros((trials, trial_dur, 2))
    for trial in range(trials):
        if np.random.randn() > 0:
            inputs[trial, :, 0] = 1.0
            targets[trial, :, 1] = s1
        else:
            targets[trial, :, 0] = s2

    # add noise to input signals
    inputs += noise * np.random.randn(trials, trial_dur, 1)

    return torch.tensor(inputs, device=device, dtype=dtype), torch.tensor(targets, device=device, dtype=dtype)


def fft_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    x_fft = torch.fft.rfft(x)
    y_fft = torch.fft.rfft(y)
    mse_real = torch.mean((torch.real(x_fft) - torch.real(y_fft))**2)
    mse_imag = torch.mean((torch.imag(x_fft) - torch.imag(y_fft)) ** 2)
    return mse_real + mse_imag + alpha*(torch.abs(torch.max(x) - 1.0) + torch.abs(torch.min(x) + 1.0))


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 1000

# input parameters
f1 = 8.0
f2 = 12.0
dt = 0.01
n_in = 1
trial_dur = 200
noise_lvl = 0.01
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)

# reservoir parameters
N = 200
n_out = 2
k = 3
sr = 0.99
bias_scale = 0.01
in_scale = 0.1
density = 0.2
out_scale = 0.1

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
n_train = 100000
n_test = 1000
init_steps = 1000
batch_size = 10
lr = 0.0001
betas = (0.9, 0.999)
alphas = (1e-2, 1e-5)

# generate inputs and targets
#############################

# get training data
x_train, y_train = get_inp(f1, f2, trial_dur, n_train, noise_lvl, dt)

# get test data
x_test, y_test = get_inp(f1, f2, trial_dur, n_test, noise_lvl, dt)

# training
##########

# initialize LR-RNN
rnn = LowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias, torch.tensor(W_z, device=device, dtype=dtype))
rnn.free_param("W")
rnn.free_param("W_z")

# set up loss function
loss_func = fft_loss

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# initial wash-out period
for step in range(init_steps):
    rnn.forward(avg_input)
y0 = rnn.y[:]

# training
current_loss = 100.0
min_loss = 1e-3
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
            loss += loss_func(outputs[:, i], trial_targ[:, i], alpha=alphas[0])

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
predictions = np.asarray(predictions).squeeze()
targets = np.asarray(targets).squeeze()
performance = np.mean((predictions - targets)**2)

# plotting
##########

# dynamics
_, axes = plt.subplots(nrows=n_out, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(f"y_{i+1}")
    if i == 1:
        ax.set_xlabel("steps")
        ax.legend()
    else:
        ax.set_title(f"Test error: {np.round(performance, decimals=2)}")
plt.tight_layout()

# trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Absolute weights: {np.round(W_abs, decimals=1)}")

plt.tight_layout()
plt.show()