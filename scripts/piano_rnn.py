import sys
path = sys.argv[-3]
sys.path.append(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import pickle

from src.rnn import RNN
from src.functions import init_weights

# function definitions
######################

def fingers_to_keys(keys: np.ndarray, n_fingers: int) -> np.ndarray:
    fingers = np.zeros((2*n_fingers,))
    diff = len(keys) - n_fingers
    if diff == 0:
        fingers[:n_fingers] = np.arange(n_fingers)
        fingers[n_fingers:] = keys.copy()
        return fingers
    keys_pressed = np.argwhere(keys > 0)[:, 0].tolist()
    idx = 0
    flip = 0
    while len(keys_pressed) > 0:
        finger_idx = n_fingers-idx if flip else idx
        key = keys_pressed.pop() if flip else keys_pressed.pop(0)
        fingers[finger_idx] = key
        fingers[n_fingers+finger_idx] = 1.0
        idx += 1
        flip = abs(flip-1)
    return fingers


def piano(n_keys: int, n_fingers: int, min_keys_pressed: int = 0, max_keys_pressed: int = None) -> tuple:

    if not max_keys_pressed:
        max_keys_pressed = n_fingers
    keys = np.arange(n_keys)

    inp, out = [], []
    for fingers in range(min_keys_pressed, max_keys_pressed + 1):
        if fingers == 0:
            inp.append(np.zeros((n_keys,)))
            out.append(np.zeros((2*n_fingers,)))
        else:
            for key_pattern in combinations(keys, r=fingers):
                inp_tmp = np.zeros((n_keys,))
                inp_tmp[np.asarray(key_pattern)] = 1.0
                inp.append(inp_tmp)
                out.append(fingers_to_keys(inp_tmp, n_fingers))

    return inp, out


def get_trials(input_patterns: list, target_patterns: list, trials: int, min_steps: int = 1, max_steps: int = 1
               ) -> tuple:
    n_patterns = len(input_patterns)
    inputs, targets = [], []
    for _ in range(trials):
        idx = np.random.choice(n_patterns)
        steps = np.random.choice(np.arange(start=min_steps, stop=max_steps))
        inputs.append(np.tile(input_patterns[idx], (steps, 1)))
        targets.append(np.tile(target_patterns[idx], (steps, 1)))
    return inputs, targets

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 200
state_vars = ["x", "y", "z"]

# piano pattern parameters
keys = int(sys.argv[-2])
fingers = int(sys.argv[-1])
noise_lvl = 0.1
min_dur = 1
max_dur = 2

# reservoir parameters
N = 200
n_in = keys
n_out = int(fingers*2)
sr = 1.1
bias_scale = 0.01
in_scale = 0.1
density = 0.2
out_scale = 0.5

# rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = torch.tensor(sr * init_weights(N, N, density), device=device, dtype=dtype)
W_r = torch.tensor(out_scale * np.random.randn(n_out, N), device=device, dtype=dtype)

# training parameters
batch_size = 100
training_trials = 500000
test_trials = 100
lr = 0.005
betas = (0.9, 0.999)
alphas = (1e-4, 1e-3)
init_steps = 1000

# generate inputs and targets
#############################

# generate all possible patterns
input_patterns, target_patterns = piano(keys, fingers)

# generate inputs and targets
X_train, y_train = get_trials(input_patterns, target_patterns, training_trials, min_steps=min_dur, max_steps=max_dur)
X_test, y_test = get_trials(input_patterns, target_patterns, test_trials, min_steps=min_dur, max_steps=max_dur)

# train RNN to generate correct key press patterns
##################################################

# initialize LR-RNN
rnn = RNN(W, W_in, bias)
rnn.free_param("W")
rnn.free_param("W_in")

# initial wash-out period
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# training
current_loss = 0.0
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(training_trials):

        trial_inp = torch.tensor(X_train[trial], device=device, dtype=dtype)
        trial_targ = torch.tensor(y_train[trial], device=device, dtype=dtype)

        for step in range(trial_inp.shape[0]):

            # get RNN output
            y = W_r @ rnn.forward(trial_inp[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))

            # calculate loss
            loss += loss_func(y, trial_targ[step])

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size
            loss += alphas[0] * torch.sum(torch.abs(rnn.W))
            loss += alphas[0] * torch.sum(torch.abs(rnn.W_in))
            loss += alphas[0] * torch.sum(torch.abs(W_r))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()
            print(f"Training phase I loss: {current_loss}")

W = rnn.W.cpu().detach().numpy()

# generate predictions
######################

# generate predictions
with torch.no_grad():
    predictions, targets = [], []
    y = W_r @ rnn.y
    for trial in range(test_trials):
        trial_inp = torch.tensor(X_test[trial], device=device, dtype=dtype)
        trial_targ = y_test[trial]
        for step in range(trial_inp.shape[0]):
            y = W_r @ rnn.forward(trial_inp[step])
            predictions.append(y.cpu().detach().numpy())
            targets.append(trial_targ[step])

# calculate MSE between predictions and targets
predictions = np.asarray(predictions)
targets = np.asarray(targets)
test_error = np.mean((predictions - targets)**2)

# saving the learned network weights
####################################

results = {"W": rnn.W, "W_in": rnn.W_in, "W_out": W_r, "bias": bias,
           "inputs": input_patterns, "targets": target_patterns}
pickle.dump(results, open(f"{path}/data/piano/piano_crnn_{keys}keys_{fingers}fingers.pkl", "wb"))

# plotting
##########

# dynamics
# fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
# ax = axes[0]
# ax.imshow(predictions[:plot_steps].T, interpolation="none", cmap="viridis", aspect="auto")
# ax.set_xlabel("steps")
# ax.set_ylabel("output channel")
# ax.set_title("Test Predictions")
# ax = axes[1]
# ax.imshow(targets[:plot_steps].T, interpolation="none", cmap="viridis", aspect="auto")
# ax.set_xlabel("steps")
# ax.set_ylabel("target channel")
# ax.set_title("Test Targets")
# plt.tight_layout()
#
# # trained weights
# fig, ax = plt.subplots(figsize=(6, 6))
# im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
# plt.colorbar(im, ax=ax)
# ax.set_xlabel("neuron")
# ax.set_ylabel("neuron")
# fig.suptitle(f"Trained Recurrent Weights")
# plt.tight_layout()
# plt.show()
