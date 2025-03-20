import sys
sys.path.append('../')
from src import LowRankOnlyRNN
import torch
import pickle
import numpy as np
from src.functions import init_weights
from scripts.task_functions import init_state, delayed_choice


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
evidence_dur = 20
response_dur = 5
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
W_in = torch.tensor(in_scale * np.random.rand(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
L = init_weights(N, k, density)
R = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(L, R))))
L *= np.sqrt(sr) / np.sqrt(sr_comb)
R *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_out, N), device=device, dtype=dtype)

# training parameters
n_train = 100000
n_test = 100
init_steps = 1000
batch_size = 20
lr = 0.001
betas = (0.9, 0.999)
alphas = (1e-5, 1e-3)

# generate inputs and targets
#############################

# get training data
x_train, y_train = delayed_choice(n_train, evidence=evidence_dur, noise=noise_lvl, device=device, dtype=dtype)

# get test data
x_test, y_test = delayed_choice(n_test, evidence=evidence_dur, noise=noise_lvl, device=device, dtype=dtype)

# training
##########

# initialize LR-RNN
rnn = LowRankOnlyRNN(W_in, bias, torch.tensor(L, device=device, dtype=dtype),
                     torch.tensor(R, device=device, dtype=dtype))
rnn.free_param("W_in")
rnn.free_param("L")
rnn.free_param("R")

# set up loss function
loss_func = torch.nn.CrossEntropyLoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# initial wash-out period
for step in range(init_steps):
    rnn.forward(avg_input)
y0 = rnn.y.detach()

# training
current_loss = 100.0
z_col = []
loss_hist = []
min_loss = 1e-3
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train):

        # set random initial condition
        y_init = init_state(y0, noise=init_noise, boundaries=(-1.0, 1.0), device=device, dtype=dtype)
        rnn.set_state(y_init, rnn.R @ y_init)

        # evidence integration period
        trial_inp = x_train[trial]
        for step in range(evidence_dur):
            rnn.forward(trial_inp[step])
            z_col.append(rnn.z.cpu().detach().numpy())

        # delay period
        for step in range(delay_dur):
            rnn.forward_a()
            z_col.append(rnn.z.cpu().detach().numpy())

        # response period
        trial_target = y_train[trial]
        for step in range(response_dur):
            y = W_r @ rnn.forward_a()
            loss += loss_func(y, trial_target)
            z_col.append(rnn.z.cpu().detach().numpy())

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size*response_dur
            loss += alphas[0] * torch.sum(torch.abs(rnn.L) @ torch.abs(rnn.R))
            loss.backward()
            current_loss = loss.item()
            loss_hist.append(current_loss)
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()

W = (rnn.L @ rnn.L).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(rnn.L)).cpu().detach().numpy())
z_col = np.asarray(z_col)

# testing
#########

# generate predictions
predictions, targets, z_test = [], [], []
with torch.no_grad():
    for trial in range(n_test):

        z_trial = []

        # set random initial condition
        y_init = init_state(y0, noise=init_noise, boundaries=(-1.0, 1.0))
        rnn.set_state(y_init, rnn.R @ y_init)

        # evidence integration period
        trial_inp = x_test[trial]
        for step in range(evidence_dur):
            rnn.forward(trial_inp[step])
            z_trial.append(rnn.z.cpu().detach().numpy())

        # delay period
        for step in range(delay_dur):
            rnn.forward_a()
            z_trial.append(rnn.z.cpu().detach().numpy())

        # response period
        trial_target = y_test[trial].cpu().detach().numpy()
        for step in range(response_dur):
            y = W_r @ rnn.forward_a()
            predictions.append(y.cpu().detach().numpy())
            targets.append(trial_target)
            z_trial.append(rnn.z.cpu().detach().numpy())

        z_test.append(np.asarray(z_trial))

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)
performance = np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))

# calculate vector field
grid_points = 20
margin = 0.0
lb = np.min(z_col, axis=0)
ub = np.max(z_col, axis=0)
width = ub - lb
coords, vf = rnn.get_vf(grid_points, lower_bounds=lb - margin*width, upper_bounds=ub + margin*width)

# save results
results = {"targets": targets, "predictions": predictions,
           "config": {"N": N, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "k": k, "alphas": alphas},
           "condition": {"repetition": rep, "noise": noise_lvl, "delay": delay_dur},
           "training_error": current_loss, "classification_performance": performance,
           "L": rnn.L.detach().cpu().numpy(), "R": rnn.R.detach().cpu().numpy(),
           "W_in": rnn.W_in.detach().cpu().numpy(), "bias": rnn.bias.detach().cpu().numpy(),
           "W_r": W_r.detach().cpu().numpy(), "vf": vf, "vf_coords": coords, "vf_sols": z_test}
pickle.dump(results, open(f"../results/lr/delayedchoice_d{int(delay_dur)}_n{int(noise_lvl*10)}_{rep}.pkl", "wb"))
