import sys
sys.path.append('../')
from src import ConceptorLowRankRNN
import torch
import pickle
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

# batch condition
k = int(sys.argv[-3])
noise_lvl = float(sys.argv[-2])
rep = int(sys.argv[-1])

# general
dtype = torch.float64
device = "cpu"

# input parameters
n_in = 2
n_train1 = 10000
n_train2 = 1000
n_test = 100
evidence_dur = 20
delay_dur = 4
response_dur = 1
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)

# reservoir parameters
N = 200
n_out = n_in
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
lam = 0.008
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
rnn = ConceptorLowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                          torch.tensor(W_z, device=device, dtype=dtype), lam, alphas[0])
rnn.init_new_conceptor(init_value="random")

# training
with torch.no_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train1):

        # initial wash-out period
        for step in range(init_steps):
            rnn.forward_c(avg_input)

        # evidence integration period
        trial_inp = x_train[trial]
        for step in range(evidence_dur):
            rnn.forward_c_adapt(trial_inp[step])

        # delay period
        for step in range(delay_dur):
            rnn.forward_c_adapt(avg_input)

# retrieve network connectivity
c = rnn.C.cpu().detach().numpy().squeeze()
W = (rnn.W @ (torch.diag(rnn.C) @ rnn.L)).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(torch.diag(rnn.C) @ rnn.L)).cpu().detach().numpy())

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

# save results
results = {"targets": targets, "predictions": predictions,
           "config": {"N": N, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "k": k, "alphas": alphas},
           "condition": {"repetition": rep, "noise": noise_lvl},
           "training_error": epsilon, "avg_weights": W_abs, "classification_performance": performance}
pickle.dump(results, open(f"../results/rfc_k{int(k)}/delayedchoice_noise{int(noise_lvl*100)}_{rep}.pkl", "wb"))
