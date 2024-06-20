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
noise_lvl = float(sys.argv[-2])
rep = int(sys.argv[-1])

# general
dtype = torch.float64
device = "cpu"

# input parameters
n_in = 2
n_train1 = 20000
n_train2 = 2000
n_test = 100
evidence_dur = 20
delay_dur = 4
response_dur = 1
noise_lvl = 2.0
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)

# reservoir parameters
N = 200
n_out = n_in
k = 3
sr = 0.99
bias_scale = 0.01
in_scale = 0.1
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
init_steps = 200
batch_size = 20
lr = 0.004
betas = (0.9, 0.999)
alphas = (1e-3, 1e-3)

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

# initialize LR-RNN
rnn = LowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias, torch.tensor(W_z, device=device, dtype=dtype))
rnn.free_param("W")
rnn.free_param("W_z")

# set up loss function
loss_func = torch.nn.CrossEntropyLoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()), lr=lr, betas=betas)

# training
current_loss = 0.0
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train1):

        # initial wash-out period
        for step in range(init_steps):
            rnn.forward(avg_input)

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
            y = W_r @ rnn.forward(avg_input)
            loss += loss_func(y, trial_target)

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size*response_dur
            loss += alphas[0] * torch.sum(torch.abs(rnn.W) @ torch.abs(rnn.W_z))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()

W = (rnn.W @ rnn.W_z).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.W) @ torch.abs(rnn.W_z)).cpu().detach().numpy())

# train final readout and generate predictions
##############################################

# harvest states
y_col, target_col = [], []
with torch.no_grad():
    for trial in range(n_train2):

        # initial wash-out period
        for step in range(init_steps):
            rnn.forward(avg_input)

        # evidence integration period
        trial_inp = x_train2[trial]
        for step in range(evidence_dur):
            rnn.forward(trial_inp[step])

        # delay period
        for step in range(delay_dur):
            rnn.forward(avg_input)

        # response period
        trial_target = y_train2[trial]
        for step in range(response_dur):
            y = rnn.forward(avg_input)
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
            y = W_r @ rnn.forward(avg_input)
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
           "training_error": epsilon, "W": (rnn.W @ rnn.W_z).cpu().detach().numpy(),
           "classification_performance": performance}
pickle.dump(results, open(f"../results/lr/delayedchoice_noise{int(noise_lvl*100)}_{rep}.pkl", "wb"))
