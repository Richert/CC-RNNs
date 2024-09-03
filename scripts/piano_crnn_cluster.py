import sys
path = sys.argv[-4]
sys.path.append(path)
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.rnn import ConceptorRNN

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
n_reps = 10

# piano pattern parameters
keys = int(sys.argv[-3])
fingers = int(sys.argv[-2])
noise_lvl = 0.1
key_dur = 1

# get parameters for cluster condition
data = pickle.load(open(f"{path}/data/piano/piano_crnn_{keys}keys_{fingers}fingers.pkl", "rb"))
cond = int(sys.argv[-1])
alpha, motifs, motif_length = data["sweep"][cond]

# matrices
W_in = data["W_in"]
bias = data["bias"]
W = data["W"]
W_r = data["W_out"]

# reservoir parameters
N = W.shape[0]
n_in = keys
n_out = int(fingers*2)

# training parameters
training_trials = 200000
test_trials = 10
loading_trials = 100000
alphas = (alpha, 1e-4)
init_steps = 1000

# get input patterns
input_patterns, target_patterns = data["inputs"], data["targets"]
n_patterns = len(input_patterns)

# results collection
results = {"alpha": alpha, "motifs": motifs, "motif_length": motif_length,
           "trial": [], "epsilon": [], "conceptors": [],
           "sequence_predictions": [], "sequence_targets": [],
           "input_predictions": [], "input_targets": []}
for n in range(n_reps):

    # generate inputs and targets
    #############################

    # generate random motifs
    motif_col, target_col = [], []
    for _ in range(motifs):
        X, y = get_trials(input_patterns, target_patterns, motif_length, key_dur)
        motif_col.append(np.asarray(X).reshape((motif_length*key_dur, n_in), order="C"))
        target_col.append(np.asarray(y).reshape((motif_length*key_dur, n_out), order="C"))

    # train RNN to generate correct key press patterns
    ##################################################

    # initialize LR-RNN
    rnn = ConceptorRNN(W, W_in, bias)

    # initial wash-out period
    avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)
    with torch.no_grad():
        for step in range(init_steps):
            rnn.forward(avg_input)

    # train conceptor for each motif
    for i, (motif, target) in enumerate(zip(motif_col, target_col)):

        motif = torch.tensor(motif, device=device, dtype=dtype)

        # collect states
        state_col = []
        with torch.no_grad():
            for trial in range(training_trials):
                for step in range(motif.shape[0]):
                    state_col.append(rnn.forward(motif[step]+noise_lvl*torch.randn((n_in,), device=device, dtype=dtype)))

        # train conceptor
        rnn.learn_conceptor(f"motif_{i}", torch.stack(state_col, dim=0), alpha=alphas[0])

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
    results["epsilon"].append(epsilon)

    # test conceptor-controlled RNN performance
    ###########################################

    conceptors, inp_predictions, inp_targets, seq_predictions, seq_targets = [], [], [], [], []
    for i, (motif, target) in enumerate(zip(motif_col, target_col)):

        rnn.activate_conceptor(f"motif_{i}")
        C = rnn.C.cpu().detach().numpy()
        conceptors.append(C)

        # collect predictions for input-driven dynamics
        with torch.no_grad():
            for trial in range(test_trials):
                for step in range(motif.shape[0]):
                    y = W_r @ rnn.forward_c(motif[step])
                    inp_predictions.append(y)
                    inp_targets.append(target[step])

        # collect predictions for sequence generation
        with torch.no_grad():
            for trial in range(test_trials):
                for step in range(motif.shape[0]):
                    y = W_r @ rnn.forward_c_a()
                    seq_predictions.append(y)
                    seq_targets.append(target[step])

    # store results
    results["conceptors"].append(conceptors)
    results["input_predictions"].append(inp_predictions)
    results["input_targets"].append(inp_targets)
    results["sequence_predictions"].append(seq_predictions)
    results["sequence_targets"].append(seq_targets)
    results["trial"].append(n)
    print(f"Finished {n+1} of {n_reps} random trials.")

# save results
pickle.dump(results, open(f"{path}/data/piano/piano_crnn_{keys}keys_{fingers}fingers_{cond}.pkl", "wb"))
