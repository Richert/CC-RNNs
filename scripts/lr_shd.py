import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py as hp
import os
from time import perf_counter
from src import LowRankRNN, init_weights


def data_generator(X, gender, speaker, n_samples: int, n_steps: int = 100, n_inp: int = 700,
                   max_time: float = 1.4, shuffle: bool = True, device: str = "cpu",
                   dtype: torch.dtype = torch.float64, start: int = None, stop: int = None):
    # get data
    gender = np.asarray(gender)
    labels = [1.0 if str(gender[idx])[2:-1] == "male" else 0.0 for idx in np.asarray(speaker)]
    firing_times = X['times']
    units_fired = X['units']

    # discretize time
    time_bins = np.linspace(0, max_time, num=n_steps)

    # choose and shuffle samples
    sample_index = np.arange(len(labels))
    if stop:
        sample_index = sample_index[:stop]
    if start:
        sample_index = sample_index[start:]
    if shuffle:
        np.random.shuffle(sample_index)

    counter = 0
    while counter < n_samples:
        idx = sample_index[counter]
        spike_times = torch.LongTensor(np.digitize(firing_times[idx], time_bins))
        inp_neurons = torch.LongTensor(np.asarray(units_fired[idx], dtype=np.int16))
        X_batch = torch.zeros([n_steps, n_inp], dtype=dtype, device=device)
        X_batch[spike_times, inp_neurons] = 1.0
        y_batch = torch.tensor([labels[idx]], dtype=torch.long, device=device)

        yield X_batch, y_batch

        counter += 1


# load data
###########

# task parameters
n_in = 700
n_steps = 100
n_out = 2

# directory where data is saved
path = "/home/rgf3807/OneDrive/data/SHD"

# load h5 file
data = hp.File(os.path.join(path, "shd_train.h5"))

# get spike representations of spoken digits and corresponding labels
X_data = data["spikes"]
gender = data["extra"]["meta_info"]["gender"]
speaker = data["extra"]["speaker"]

# digit labels
labels = ["male", "female"]

# plot training samples
n_samples = 9
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 7))
single_axes = [axes[0, 0], axes[0, 1], axes[0, 2],
               axes[1, 0], axes[1, 1], axes[1, 2],
               axes[2, 0], axes[2, 1], axes[2, 2]]
for (X, y), ax in zip(data_generator(X_data, gender, speaker, n_samples, n_steps=n_steps, n_inp=n_in), single_axes):
    X = X.cpu().detach().numpy()
    ax.imshow(X.T, aspect="auto", interpolation="none", cmap="Greys")
    idx = y.cpu().detach().numpy()[0]
    ax.set_title(f"Gender: {labels[idx]}")
    ax.set_xlabel("time step")
    ax.set_ylabel("input neuron")
fig.suptitle("Training samples")
plt.tight_layout()

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 4000

# reservoir parameters
N = 200
k = 6
sr = 1.1
bias_scale = 0.01
in_scale = 0.015
density = 0.2
out_scale = 0.5

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
n_epochs = 300
n_samples = 10
init_steps = int(0.5*n_steps)
test_samples = 100
lr = 0.001
betas = (0.9, 0.999)
alphas = (0.1, 1e-3)

# model training
################

# initialize LR-RNN
rnn = LowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias, torch.tensor(W_z, device=device, dtype=dtype))
rnn.free_param("W")
rnn.free_param("W_z")
activation_func = torch.nn.Softmax()

# initial wash-out period
avg_input = torch.zeros((n_in,), device=device, dtype=dtype)

# set up loss function
loss_func = torch.nn.CrossEntropyLoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# training epochs
epoch_loss = []
for epoch in range(n_epochs):

    t0 = perf_counter()
    loss = torch.zeros((1,))
    for sample, (X, y_t) in enumerate(
            data_generator(X_data, gender, speaker, n_samples,
                           n_steps=n_steps, n_inp=n_in, device=device, dtype=dtype, shuffle=True)
    ):

        for step in range(init_steps):
            rnn.forward(avg_input)

        for step in range(n_steps):

            # get RNN output
            rnn.forward(X[step, :])

            # calculate loss
            y = activation_func(W_r @ rnn.y)
            loss += loss_func(y, y_t[0])

    optim.zero_grad()
    loss += alphas[0] * torch.sum(torch.abs(rnn.W) @ torch.abs(rnn.L))
    loss += alphas[1] * torch.sum(torch.abs(W_r))
    loss.backward()
    epoch_loss.append(loss.item())
    optim.step()
    rnn.detach()
    print(f"Loss after epoch {epoch + 1}: {epoch_loss[-1]}")

# get trained weights
W = (rnn.W @ rnn.L).cpu().detach().numpy()
lambdas = np.abs(np.linalg.eigvals(W))
sr_trained = np.max(lambdas)

# plot trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Trained SR: {np.round(sr_trained, decimals=1)}")
plt.tight_layout()

# plot training performance
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(epoch_loss)
ax.set_xlabel("epoch")
ax.set_ylabel("CE")
ax.set_title("Cross-Entropy Loss across Training Epochs")
plt.tight_layout()

# model testing
###############

# load test data
data = hp.File(os.path.join(path, "shd_test.h5"))
X_data = data["spikes"]
gender = data["extra"]["meta_info"]["gender"]
speaker = data["extra"]["speaker"]

# get model predictions
test_loss, predictions, targets, dynamics = [], [], [], []
with torch.no_grad():
    for X, y_t in data_generator(X_data, gender, speaker, test_samples,
                                 n_steps=n_steps, n_inp=n_in, device=device, dtype=dtype):

        # get initial state
        for step in range(init_steps):
            rnn.forward(avg_input)
            dynamics.append(rnn.y.cpu().detach().numpy())

        for step in range(n_steps):

            rnn.forward(X[step, :])

            # get prediction
            y = activation_func(W_r @ rnn.y)

            # calculate loss
            loss = loss_func(y, y_t[0])
            predictions.append(y.cpu().detach().numpy())
            target = np.zeros((n_out,))
            target[y_t.item()] = 1.0
            targets.append(target)

            # collect loss
            test_loss.append(loss.item())

            # collect dynamics
            dynamics.append(rnn.y.cpu().detach().numpy())

targets = np.asarray(targets)
predictions = np.asarray(predictions)
performance = np.mean(np.argmax(targets, axis=1) == np.argmax(predictions, axis=1))
dynamics = np.asarray(dynamics)


# plot test performance
_, ax = plt.subplots(figsize=(12, 4))
ax.plot(test_loss)
ax.set_xlabel("sample")
ax.set_ylabel("CE")
ax.set_title("Cross-Entropy Loss on Test Samples")
plt.tight_layout()

# plot predictions and targets
fig, axes = plt.subplots(nrows=n_out, figsize=(12, 2 * n_out))
for i in range(n_out):
    ax = axes[i]
    ax.plot(targets[:, i], label="targets")
    ax.plot(predictions[:, i], label="predictions")
    ax.set_xlabel("time step")
    ax.set_ylabel("probability")
    ax.set_title(f"Speaker class: {labels[i]}")
fig.suptitle(f"Average classification performance: {performance}")
plt.tight_layout()

# plot network dynamics
_, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(dynamics.T, aspect="auto", interpolation="none", cmap="viridis", vmax=1.0, vmin=-1.0)
plt.colorbar(im, ax=ax, shrink=0.7)
ax.set_xlabel("steps")
ax.set_ylabel("neuron")
ax.set_title("Network dynamics on test samples")
plt.tight_layout()
plt.show()
