import sys
sys.path.append('../')
from src import LowRankRNN
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
delay_dur = int(sys.argv[-3])
noise_lvl = float(sys.argv[-2])
rep = int(sys.argv[-1])

# general
dtype = torch.float64
device = "cpu"

# input parameters
n_in = 2
n_train = 50000
n_test = 100
evidence_dur = 20
response_dur = 2
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
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_out, N), device=device, dtype=dtype)

# training parameters
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
rnn = LowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias, torch.tensor(W_z, device=device, dtype=dtype))
rnn.free_param("W")
rnn.free_param("W_z")
f = torch.nn.Softmax(dim=0)

# set up loss function
loss_func = torch.nn.CrossEntropyLoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# initial wash-out period
for step in range(init_steps):
    rnn.forward(avg_input)
y0 = rnn.y[:]

# training
current_loss = 0.0
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train):

        rnn.y = y0 + init_noise * torch.randn(N)

        # evidence integration period
        trial_inp = x_train[trial]
        for step in range(evidence_dur):
            rnn.forward(trial_inp[step])

        # delay period
        for step in range(delay_dur):
            rnn.forward(avg_input)

        # response period
        trial_target = y_train[trial]
        for step in range(response_dur):
            y = f(W_r @ rnn.forward(avg_input))
            loss += loss_func(y, trial_target)

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size*response_dur
            loss += alphas[0] * torch.sum(torch.abs(rnn.W) @ torch.abs(rnn.L))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()

W = (rnn.W @ rnn.L).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(rnn.L)).cpu().detach().numpy())

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

        # delay period
        for step in range(delay_dur):
            rnn.forward(avg_input)

        # response period
        trial_target = y_test[trial].cpu().detach().numpy()
        for step in range(response_dur):
            y = f(W_r @ rnn.forward(avg_input))
            predictions.append(y.cpu().detach().numpy())
            targets.append(trial_target)

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)
performance = np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))

# save results
results = {"targets": targets, "predictions": predictions,
           "config": {"N": N, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "k": k, "alphas": alphas},
           "condition": {"repetition": rep, "noise": noise_lvl, "delay": delay_dur},
           "training_error": current_loss, "avg_weights": W_abs,
           "classification_performance": performance}
pickle.dump(results, open(f"../results/lr/delayedchoice_d{int(delay_dur)}_n{int(noise_lvl*10)}_{rep}.pkl", "wb"))
