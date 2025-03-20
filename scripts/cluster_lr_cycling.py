import sys
sys.path.append('../')
from src import LowRankOnlyRNN
import torch
import numpy as np
from src.functions import init_weights
from scripts.task_functions import cycling
import pickle

# parameter definition
######################

# batch condition
inp_noise = int(sys.argv[-3])
init_noise = float(sys.argv[-2])
rep = int(sys.argv[-1])

# general
dtype = torch.float64
device = "cpu"
plot_steps = 1000

# input parameters
freq = 5.0
dt = 0.01
n_in = 2
trial_dur = 400
min_cycling_dur = 50
inp_dur = 5
inp_damping = 1.0
padding = int(0.2*trial_dur)

# reservoir parameters
N = 200
n_out = 1
k = 2
sr = 0.99
bias_scale = 0.01
in_scale = 0.1
density = 0.2
out_scale = 0.1

# rnn matrices
W_in = torch.tensor(in_scale * np.random.rand(N, n_in), device=device, dtype=dtype, requires_grad=False)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype, requires_grad=False)
L = init_weights(N, k, density)
R = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(L, R))))
L *= np.sqrt(sr) / np.sqrt(sr_comb)
R *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_out, N), device=device, dtype=dtype)

# training parameters
n_train = 100000
n_test = 1000
init_steps = 1000
batch_size = 20
lr = 0.0005
betas = (0.9, 0.999)
alphas = (1e-5,)

# generate inputs and targets
#############################

# get training data
x_train, y_train = cycling(freq, trial_dur, min_cycling_dur, inp_dur, inp_damping, n_train, inp_noise, dt,
                           device=device, dtype=dtype)

# get test data
x_test, y_test = cycling(freq, trial_dur, min_cycling_dur, inp_dur, inp_damping, n_test, inp_noise, dt,
                         device=device, dtype=dtype)

# training
##########

# initialize LR-RNN
rnn = LowRankOnlyRNN(W_in, bias, torch.tensor(L, device=device, dtype=dtype),
                     torch.tensor(R, device=device, dtype=dtype))
rnn.free_param("W_in")
rnn.free_param("L")
rnn.free_param("R")
y0 = rnn.y.detach()
z0 = rnn.z.detach()

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# initial wash-out period
for step in range(init_steps):
    rnn.forward_a()
y0 = rnn.y.detach()
z0 = rnn.z.detach()

# training
current_loss = 100.0
min_loss = 1e-3
loss_hist = []
z_col = []
with torch.enable_grad():

    loss = torch.zeros((1,))
    for trial in range(n_train):

        # wash out
        rnn.set_state(y0, z0)
        for step in range(init_steps):
            rnn.forward(init_noise * torch.randn(n_in, dtype=dtype, device=device)).cpu().detach().numpy()

        # trial
        trial_inp = x_train[trial]
        trial_targ = y_train[trial]
        for step in range(trial_dur):
            y = W_r @ rnn.forward(trial_inp[step])
            loss += loss_func(y, trial_targ[step])
            z_col.append(rnn.z.detach().cpu().numpy())

        # make update
        if (trial + 1) % batch_size == 0:
            optim.zero_grad()
            loss /= batch_size*trial_dur
            loss += alphas[0] * torch.sum(torch.abs(rnn.L) @ torch.abs(rnn.R))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()
            loss_hist.append(current_loss)

        if current_loss < min_loss:
            break

W = (rnn.L @ rnn.R).cpu().detach().numpy() + rnn.L.cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(rnn.R)).cpu().detach().numpy())
z_col = np.asarray(z_col)

# testing
#########

# generate predictions
predictions, targets = [], []
test_loss = torch.zeros((1,))
z_test = []
with torch.no_grad():
    for trial in range(n_test):

        # wash out
        rnn.set_state(y0, z0)
        for step in range(init_steps):
            rnn.forward(init_noise * torch.randn(n_in, dtype=dtype, device=device)).cpu().detach().numpy()

        # trial
        trial_inp = x_test[trial]
        trial_targ = y_test[trial]
        trial_predictions = []
        trial_z = []
        for step in range(trial_dur):
            y = W_r @ rnn.forward(trial_inp[step])
            test_loss += loss_func(y, trial_targ[step])
            trial_predictions.append(y)
            targets.append(trial_targ[step])
            trial_z.append(rnn.z.detach().cpu().numpy())
        z_test.append(np.asarray(trial_z))
        trial_predictions = torch.stack(trial_predictions, dim=0)
        predictions.extend(trial_predictions.cpu().detach().numpy().tolist())

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)

test_loss = test_loss.item() / (n_test * trial_dur)

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
           "condition": {"repetition": rep, "inp_noise": inp_noise, "init_noise": init_noise},
           "training_error": current_loss, "L": rnn.L.detach().cpu().numpy(), "R": rnn.R.detach().cpu().numpy(),
           "W_in": rnn.W_in.detach().cpu().numpy(), "bias": rnn.bias.detach().cpu().numpy(),
           "W_r": W_r.detach().cpu().numpy(), "vf": vf, "vf_coords": coords, "vf_sols": z_test}
pickle.dump(results, open(f"../results/lr/cycling_inp{int(inp_noise*10.0)}_init{int(init_noise*10)}_{rep}.pkl", "wb"))
