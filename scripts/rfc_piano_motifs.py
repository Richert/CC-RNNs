from src.rnn import ConceptorLowRankOnlyRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

# function definitions
######################

def get_trials(input_patterns: list, target_patterns: list, trials: int, trial_steps: int) -> tuple:
    n_patterns = len(input_patterns)
    inputs, targets = [], []
    for _ in range(trials):
        idx = np.random.choice(n_patterns)
        inputs.append(np.tile(input_patterns[idx], (trial_steps, 1)))
        targets.append(np.tile(target_patterns[idx], (trial_steps, 1)))
    return inputs, targets

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 200
state_vars = ["x", "y", "z"]

# piano pattern parameters
keys = 4
fingers = 2
noise_lvl = 0.1
motifs = 3
motif_length = 4
key_dur = 1

# rnn matrices
matrices = pickle.load(open("../data/lr_piano_weights.pkl", "rb"))
W_in = matrices["W_in"]
bias = matrices["bias"]
L = matrices["L"]
R = matrices["R"]
W_r = matrices["W_out"]

# reservoir parameters
N = L.shape[0]
k = L.shape[1]
n_in = keys
n_out = int(fingers*2)

# training parameters
training_trials = 500000
test_trials = 10
loading_trials = 100000
alphas = (80.0, 1e-4)
lam = 0.002
init_steps = 1000

# generate inputs and targets
#############################

# generate all possible patterns
input_patterns, target_patterns = matrices["inputs"], matrices["targets"]
n_patterns = len(input_patterns)

# generate random motifs
motif_col, target_col = [], []
for _ in range(motifs):
    X, y = get_trials(input_patterns, target_patterns, motif_length, key_dur)
    motif_col.append(np.asarray(X).reshape((motif_length*key_dur, n_in), order="C"))
    target_col.append(np.asarray(y).reshape((motif_length*key_dur, n_out), order="C"))

# train RNN to generate correct key press patterns
##################################################

# initialize LR-RNN
rnn = ConceptorLowRankOnlyRNN(W_in, bias, L, R, lam=lam, alpha=alphas[0])

# initial wash-out period
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward(avg_input)

# train conceptor for each motif
for i, (motif, target) in enumerate(zip(motif_col, target_col)):

    motif = torch.tensor(motif, device=device, dtype=dtype)
    rnn.init_new_conceptor("random")

    # train conceptor
    state_col = []
    with torch.no_grad():
        for trial in range(training_trials):
            for step in range(motif.shape[0]):
                rnn.forward_c_adapt(motif[step]+noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))
    rnn.store_conceptor(f"motif_{i}")

# input loading
state_col, input_col = [], []
for i, (motif, target) in enumerate(zip(motif_col, target_col)):
    with torch.no_grad():
        motif = torch.tensor(motif, device=device, dtype=dtype)
        rnn.activate_conceptor(f"motif_{i}")
        for trial in range(loading_trials):
            for step in range(motif.shape[0]):
                state_col.append(rnn.y)
                input_col.append(motif[step])
                rnn.forward_c(motif[step]+noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))
_, epsilon = rnn.load_input(torch.stack(state_col, dim=0).T, torch.stack(input_col, dim=0).T, tychinov_alpha=alphas[1])
print(f"Input loading error: {epsilon}")

# plot conceptor-controlled RNN performance
###########################################

fig, axes = plt.subplots(nrows=2*motifs, figsize=(12, 4*motifs))

for i, (motif, target) in enumerate(zip(motif_col, target_col)):

    rnn.activate_conceptor(f"motif_{i}")
    C = rnn.c_weights.cpu().detach().numpy()

    # collect states
    predictions, targets = [], []
    with torch.no_grad():
        for trial in range(test_trials):
            for step in range(motif.shape[0]):
                y = W_r @ rnn.forward_c_a()
                predictions.append(y)
                targets.append(target[step])

    # plotting
    ax = axes[2*i]
    ax.set_title(f"Motif {i}: Conceptor dimension = {np.sum(C)}")
    ax.imshow(np.asarray(predictions).T, aspect="auto", interpolation="none", cmap="viridis")
    ax = axes[2*i+1]
    ax.imshow(np.asarray(targets).T, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_xlabel("steps")

plt.tight_layout()
plt.show()
