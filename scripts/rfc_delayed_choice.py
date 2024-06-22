from src import RandomFeatureConceptorRNN
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
n_train1 = 20000
n_train2 = 2000
n_test = 100
evidence_dur = 20
delay_dur = 20
response_dur = 1
noise_lvl = 0.1
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)

# reservoir parameters
N = 200
n_out = n_in
k = 50
sr = 1.2
bias_scale = 0.01
in_scale = 0.1
density = 0.2

# rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)

# training parameters
init_steps = 200
batch_size = 20
lam = 0.002
alphas = (15.0, 1e-3)

# generate inputs and targets
#############################

# get training data
x_train, y_train = two_choice(n_train1, evidence=evidence_dur, noise=noise_lvl)

# get readout training data
x_train2, y_train2 = two_choice(n_train2, evidence=evidence_dur, noise=noise_lvl)

# get test data
x_test, y_test = two_choice(n_test, evidence=evidence_dur, noise=noise_lvl)

# train low-rank RNN to predict next time step of Lorenz attractor
##################################################################

# initialize RFC-RNN
rnn = RandomFeatureConceptorRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                                torch.tensor(W_z, device=device, dtype=dtype), lam, alphas[0])
rnn.init_new_conceptor(init_value="random")

# training
# with torch.no_grad():
#
#     loss = torch.zeros((1,))
#     for trial in range(n_train1):
#
#         # initial wash-out period
#         for step in range(init_steps):
#             rnn.forward_c(avg_input)
#
#         # evidence integration period
#         trial_inp = x_train[trial]
#         for step in range(evidence_dur):
#             rnn.forward_c_adapt(trial_inp[step])
#
#         # delay period
#         for step in range(delay_dur):
#             rnn.forward_c_adapt(avg_input)

# retrieve network connectivity
c = rnn.C.cpu().detach().numpy().squeeze()
W = (rnn.W @ (torch.diag(rnn.C) @ rnn.W_z)).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(torch.diag(rnn.C) @ rnn.W_z)).cpu().detach().numpy())
print(f"Conceptor: {np.sum(c)}")

# train final readout and generate predictions
##############################################

# harvest states
y_col, target_col = [], []
with torch.no_grad():
    for trial in range(n_train2):

        # initial wash-out period
        for step in range(init_steps):
            rnn.forward_c(avg_input)

        # evidence integration period
        trial_inp = x_train2[trial]
        for step in range(evidence_dur):
            rnn.forward_c(trial_inp[step])

        # delay period
        for step in range(delay_dur):
            rnn.forward_c(avg_input)

        # response period
        trial_target = y_train2[trial]
        for step in range(response_dur):
            y = rnn.forward_c(avg_input)
            y_col.append(y)
            target_col.append(trial_target)

# train readout
y_col = torch.stack(y_col, dim=0)
target_col = torch.stack(target_col, dim=0)
W_r, epsilon = rnn.train_readout(y_col.T, target_col.T, alphas[1])
print(f"Readout training error: {float(torch.mean(epsilon).cpu().detach().numpy())}")

# generate predictions
predictions, targets = [], []
with torch.no_grad():
    for trial in range(n_test):

        # initial wash-out period
        for step in range(init_steps):
            rnn.forward_c(avg_input)

        # evidence integration period
        trial_inp = x_test[trial]
        for step in range(evidence_dur):
            rnn.forward_c(trial_inp[step])

        # delay period
        for step in range(delay_dur):
            rnn.forward_c(avg_input)

        # response period
        trial_target = y_test[trial].cpu().detach().numpy()
        for step in range(response_dur):
            y = W_r @ rnn.forward_c(avg_input)
            predictions.append(y.cpu().detach().numpy())
            targets.append(trial_target)

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)
performance = np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))

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
plt.show()
