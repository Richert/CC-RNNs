from src import LowRankRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights


# function definitions
######################

def two_choice(trials: int, evidence: int, noise: float) -> tuple:

    # allocate arrays
    inputs = torch.randn(trials, evidence, 2, device=device, dtype=dtype) * noise
    targets = torch.zeros((trials, 2), device=device, dtype=dtype)

    # create inputs and targets
    for trial in range(trials):

        # choose random input channel
        channel = np.random.randint(low=0, high=2)
        inputs[trial, :, channel] += 1.0
        targets[trial, channel] = 1.0

    return inputs, targets


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 100

# input parameters
n_in = 2
evidence_dur = 20
delay_dur = 15
response_dur = 2
noise_lvl = 0.2
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)

# reservoir parameters
N = 200
n_out = n_in
k = 2
sr = 1.1
bias_scale = 0.01
in_scale = 0.1
density = 0.2
out_scale = 0.5
init_noise = 0.003

# rnn matrices
lbd = 0.99
W_in = torch.tensor(in_scale * np.random.rand(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = torch.tensor((1-lbd)*sr*init_weights(N, N, density), device=device, dtype=dtype)
L = init_weights(N, k, density)
R = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(L, R))))
L *= np.sqrt(lbd*sr) / np.sqrt(sr_comb)
R *= np.sqrt(lbd*sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_out, N), device=device, dtype=dtype)

# training parameters
n_train = 5000
n_test = 100
init_steps = 1000
batch_size = 20
lr = 0.001
betas = (0.9, 0.999)
alphas = (1e-5, 1e-3)

# generate inputs and targets
#############################

# get training data
x_train, y_train = two_choice(n_train, evidence=evidence_dur, noise=noise_lvl)

# get test data
x_test, y_test = two_choice(n_test, evidence=evidence_dur, noise=noise_lvl)

# training
##########

# initialize LR-RNN
rnn = LowRankRNN(W, W_in, bias, torch.tensor(L, device=device, dtype=dtype),
                 torch.tensor(R, device=device, dtype=dtype))
rnn.free_param("L")
rnn.free_param("R")
f = torch.nn.Softmax(dim=0)

# set up loss function
loss_func = torch.nn.CrossEntropyLoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# initial wash-out period
for step in range(init_steps):
    rnn.forward(avg_input)
y0 = rnn.y.detach()

# training
current_loss = 0.0
z_col = []
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train):

        rnn.y = y0 + init_noise * torch.randn(N)

        # evidence integration period
        trial_inp = x_train[trial]
        for step in range(evidence_dur):
            rnn.forward(trial_inp[step])
            z_col.append(rnn.z.cpu().detach().numpy())

        # delay period
        for step in range(delay_dur):
            rnn.forward(avg_input)
            z_col.append(rnn.z.cpu().detach().numpy())

        # response period
        trial_target = y_train[trial]
        for step in range(response_dur):
            y = f(W_r @ rnn.forward(avg_input))
            loss += loss_func(y, trial_target)
            z_col.append(rnn.z.cpu().detach().numpy())

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size*response_dur
            loss += alphas[0] * torch.sum(torch.abs(rnn.L) @ torch.abs(rnn.R))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()
            print(f"Training phase I loss: {current_loss}")

W = (rnn.W + rnn.L @ rnn.R).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(rnn.R)).cpu().detach().numpy())

# testing
#########

# generate predictions
predictions, targets = [], []
with torch.no_grad():
    for trial in range(n_test):

        # initial wash-out period
        for step in range(init_steps):
            rnn.forward(avg_input)

        # evidence integration period
        trial_inp = x_test[trial]
        for step in range(evidence_dur):
            rnn.forward(trial_inp[step])
            z_col.append(rnn.z.cpu().detach().numpy())

        # delay period
        for step in range(delay_dur):
            rnn.forward(avg_input)
            z_col.append(rnn.z.cpu().detach().numpy())

        # response period
        trial_target = y_test[trial].cpu().detach().numpy()
        for step in range(response_dur):
            y = f(W_r @ rnn.forward(avg_input))
            predictions.append(y.cpu().detach().numpy())
            targets.append(trial_target)
            z_col.append(rnn.z.cpu().detach().numpy())

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)
z_col = np.asarray(z_col)
performance = np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))

# calculate vector field
grid_points = 20
lb = np.min(z_col, axis=0)
ub = np.max(z_col, axis=0)
coords, vf = rnn.get_vf(grid_points, lower_bounds=lb, upper_bounds=ub)

# plotting
##########

# dynamics
fig, axes = plt.subplots(nrows=n_out, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(f"Target class {i+1}")
    if i == n_out-1:
        ax.set_xlabel("steps")
        ax.legend()
fig.suptitle(f"Classification performance: {np.round(performance, decimals=2)}")
plt.tight_layout()

# trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Absolute weights: {np.round(W_abs, decimals=1)}")
plt.tight_layout()

# vector field
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(coords[:, 0], coords[:, 1], vf[:, 0], vf[:, 1])
ax.set_xlabel("z_1")
ax.set_ylabel("z_2")
ax.set_title("Vectorfield")
plt.tight_layout()

plt.show()
