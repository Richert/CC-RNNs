from src import LowRankRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights


# function definitions
######################

def get_inp(freq: float, trial_dur: int, min_cycling_dur: int, inp_dur: int, trials: int, noise: float, dt: float) -> tuple:

    # create inputs and targets
    inputs = np.zeros((trials, trial_dur, 2))
    targets = np.zeros((trials, trial_dur, 1))
    for trial in range(trials):
        start = np.random.randint(low=0, high=int(0.5*(trial_dur-min_cycling_dur)))
        stop = np.random.randint(low=start+min_cycling_dur, high=trial_dur-inp_dur)
        inputs[trial, start:start+inp_dur, 0] = 1.0
        inputs[trial, stop:stop+inp_dur, 1] = -1.0
        targets[trial, start:stop, 0] = np.sin(2.0*np.pi*freq*np.linspace(0.0, (stop-start)*dt, stop-start))

    # add noise to input signals
    inputs += noise * np.random.randn(trials, trial_dur, 1)

    return torch.tensor(inputs, device=device, dtype=dtype), torch.tensor(targets, device=device, dtype=dtype)


def cc_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, padding: int = 1) -> torch.Tensor:
    cc = torch.abs(torch.nn.functional.conv1d(x[None, None, :], y[None, None, :], padding=padding))
    return -torch.max(cc) + alpha*(torch.abs(torch.max(x) - 1.0) + torch.abs(torch.min(x) + 1.0))


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 1000

# input parameters
freq = 5.0
dt = 0.01
n_in = 2
trial_dur = 200
min_cycling_dur = 50
inp_dur = 5
padding = int(0.2*trial_dur)
inp_noise = 0.1

# reservoir parameters
N = 200
n_out = 1
k = 3
sr = 1.05
bias_scale = 0.01
in_scale = 0.2
density = 0.2
out_scale = 0.1
init_noise = 0.05

# rnn matrices
lbd = 0.99
W_in = torch.tensor(in_scale * np.random.rand(N, n_in), device=device, dtype=dtype, requires_grad=False)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype, requires_grad=False)
W = torch.tensor((1-lbd)*sr*init_weights(N, N, density), device=device, dtype=dtype, requires_grad=False)
L = init_weights(N, k, density)
R = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(L, R))))
L *= np.sqrt(sr*lbd) / np.sqrt(sr_comb)
R *= np.sqrt(sr*lbd) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_out, N), device=device, dtype=dtype)

# training parameters
n_train = 500
n_test = 1000
init_steps = 100
batch_size = 3
lr = 0.0005
betas = (0.9, 0.999)
alphas = (1.0, 1e-4)

# generate inputs and targets
#############################

# get training data
x_train, y_train = get_inp(freq, trial_dur, min_cycling_dur, inp_dur, n_train, inp_noise, dt)

# get test data
x_test, y_test = get_inp(freq, trial_dur, min_cycling_dur, inp_dur, n_test, inp_noise, dt)

# training
##########

# initialize LR-RNN
rnn = LowRankRNN(W, W_in, bias, torch.tensor(L, device=device, dtype=dtype),
                 torch.tensor(R, device=device, dtype=dtype))
rnn.free_param("W_in")
rnn.free_param("L")
rnn.free_param("R")

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# training
current_loss = 100.0
min_loss = -50.0
loss_hist = []
z_col = []
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train):

        # wash out
        for step in range(init_steps):
            rnn.forward(init_noise * torch.randn(n_in, dtype=dtype, device=device)).cpu().detach().numpy()

        # trial
        trial_inp = x_train[trial]
        trial_targ = y_train[trial]
        for step in range(trial_dur):
            y = W_r @ rnn.forward(trial_inp[step])
            loss += loss_func(y, trial_targ[step])
            z_col.append(rnn.z.detach().cpu().numpy())

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size*trial_dur
            loss += alphas[1] * torch.sum(torch.abs(rnn.L) @ torch.abs(rnn.R))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()
            loss_hist.append(current_loss)
            print(f"MSE loss after {trial+1} training trials: {current_loss}")

        if current_loss < min_loss:
            break

W = (rnn.L @ rnn.R).cpu().detach().numpy() + rnn.W.cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(rnn.R)).cpu().detach().numpy())

# testing
#########

# generate predictions
predictions, targets = [], []
test_loss = 0.0
with torch.no_grad():
    for trial in range(n_test):

        # wash out
        for step in range(init_steps):
            rnn.forward(init_noise * torch.randn(n_in, dtype=dtype, device=device)).cpu().detach().numpy()

        # trial
        trial_inp = x_test[trial]
        trial_targ = y_test[trial]
        trial_predictions = []
        for step in range(trial_dur):
            y = W_r @ rnn.forward(trial_inp[step])
            trial_predictions.append(y)
            targets.append(trial_targ[step])
            z_col.append(rnn.z.detach().cpu().numpy())
        trial_predictions = torch.stack(trial_predictions, dim=0)
        predictions.extend(trial_predictions.cpu().detach().numpy().tolist())
        for i in range(n_out):
            test_loss += cc_loss(trial_predictions[:, 0], trial_targ[:, 0], alpha=alphas[0], padding=padding)

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)
z_col = np.asarray(z_col)
performance = test_loss / n_test

# calculate vector field
grid_points = 20
lb = np.min(z_col, axis=0)
ub = np.max(z_col, axis=0)
coords, vf = rnn.get_vf(grid_points, lower_bounds=lb, upper_bounds=ub)

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
fig.suptitle(f"Summed LR connectivity: {np.round(W_abs, decimals=1)}")

# loss history
_, ax = plt.subplots(figsize=(10, 3))
ax.plot(loss_hist)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.tight_layout()

# vector field
_, ax = plt.subplots(figsize=(8, 8))
ax.quiver(coords[:, 0], coords[:, 1], vf[:, 0], vf[:, 1])
ax.set_xlabel("z_1")
ax.set_ylabel("z_2")
ax.set_title("Vectorfield 1")
plt.tight_layout()
_, ax = plt.subplots(figsize=(8, 8))
ax.quiver(coords[:, 1], coords[:, 2], vf[:, 1], vf[:, 2])
ax.set_xlabel("z_2")
ax.set_ylabel("z_3")
ax.set_title("Vectorfield 2")
plt.tight_layout()
_, ax = plt.subplots(figsize=(8, 8))
ax.quiver(coords[:, 0], coords[:, 2], vf[:, 0], vf[:, 2])
ax.set_xlabel("z_1")
ax.set_ylabel("z_3")
ax.set_title("Vectorfield 3")
plt.tight_layout()
plt.show()
