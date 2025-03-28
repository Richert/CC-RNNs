from src import LowRankOnlyRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights
from scripts.task_functions import init_state, feature_choice

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 500

# input parameters
n_features = 2
n_in = n_features + 1
target_feature = 0
evidence_dur = 20
delay_min = 10
delay_max = 20
response_dur = 10
noise_lvl = 0.1
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)

# reservoir parameters
N = 200
n_out = 1
k = 2
sr = 1.1
bias_scale = 0.01
in_scale = 0.1
density = 0.2
out_scale = 0.2
init_noise = 0.2

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
n_train = 50000
n_test = 100
init_steps = 1000
batch_size = 20
lr = 0.001
betas = (0.9, 0.999)
alphas = (1e-5, 1e-3)

# generate inputs and targets
#############################

# get training data
x_train, y_train = feature_choice(n_features, target_feature, n_train,
                                  evidence=evidence_dur, delay_min=delay_min, delay_max=delay_max,
                                  response=response_dur, noise=noise_lvl)

# get test data
x_test, y_test = feature_choice(n_features, target_feature, n_test,
                                evidence=evidence_dur, delay_min=delay_min, delay_max=delay_max,
                                response=response_dur, noise=noise_lvl)

# training
##########

# initialize LR-RNN
rnn = LowRankOnlyRNN(W_in, bias, torch.tensor(L, device=device, dtype=dtype),
                     torch.tensor(R, device=device, dtype=dtype))
# rnn.free_param("W_in")
rnn.free_param("L")
rnn.free_param("R")
f = torch.nn.Tanh()

# set up loss function
loss_func = torch.nn.MSELoss()

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

        # trial
        trial_inp = torch.tensor(x_train[trial], device=device, dtype=dtype)
        trial_out = torch.tensor(y_train[trial], device=device, dtype=dtype)
        for step in range(trial_inp.shape[0]):
            y = f(W_r @ rnn.forward(trial_inp[step]))
            loss += loss_func(y, trial_out[step])
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
            print(f"Training loss after {trial+1} trials: {current_loss}")

        if current_loss < min_loss:
            break

W = (rnn.L + rnn.L @ rnn.R).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(rnn.R)).cpu().detach().numpy())
z_col = np.asarray(z_col)

# testing
#########

# generate predictions
predictions, targets, z_test = [], [], []
test_loss = torch.zeros((1,))
with torch.no_grad():
    for trial in range(n_test):

        z_trial = []

        # set random initial condition
        y_init = init_state(y0, noise=init_noise, boundaries=(-1.0, 1.0))
        rnn.set_state(y_init, rnn.R @ y_init)

        # trial
        trial_inp = torch.tensor(x_train[trial], device=device, dtype=dtype)
        trial_out = torch.tensor(y_train[trial], device=device, dtype=dtype)
        for step in range(trial_inp.shape[0]):
            y = f(W_r @ rnn.forward(trial_inp[step]))
            test_loss += loss_func(y, trial_out[step])
            predictions.append(y.cpu().detach().numpy())
            targets.append(trial_out[step].cpu().detach().numpy())
            z_trial.append(rnn.z.cpu().detach().numpy())

        z_test.append(np.asarray(z_trial))

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)
test_loss = test_loss.item()

# calculate vector field
grid_points = 20
margin = 0.0
lb = np.min(z_col, axis=0)
ub = np.max(z_col, axis=0)
width = ub - lb
coords, vf = rnn.get_vf(grid_points, lower_bounds=lb - margin*width, upper_bounds=ub + margin*width)

# plotting
##########

# dynamics
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(targets[:plot_steps, 0], color="royalblue", label="target")
ax.plot(predictions[:plot_steps, 0], color="darkorange", label="prediction")
ax.set_ylabel(f"Prediction on test data")
ax.set_xlabel("steps")
ax.legend()
fig.suptitle(f"Test loss: {np.round(test_loss, decimals=2)}")
plt.tight_layout()

# trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Absolute weights: {np.round(W_abs, decimals=1)}")
plt.tight_layout()

# loss history
_, ax = plt.subplots(figsize=(10, 3))
ax.plot(loss_hist)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.tight_layout()

# vector field
plot_trials = 8
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(coords[:, 0], coords[:, 1], vf[:, 0], vf[:, 1])
for _ in range(plot_trials):
    idx = np.random.randint(0, n_test)
    trial_z = z_test[idx]
    l = ax.plot(trial_z[:evidence_dur, 0], trial_z[:evidence_dur, 1], linestyle="solid")
    ax.plot(trial_z[evidence_dur-1:, 0], trial_z[evidence_dur-1:, 1], linestyle="dotted", c=l[0].get_color())
    ax.scatter(trial_z[0, 0], trial_z[0, 1], marker="o", s=50.0, c=l[0].get_color())
    ax.scatter(trial_z[-1, 0], trial_z[-1, 1], marker="x", s=50.0, c=l[0].get_color())
ax.set_xlabel("z_1")
ax.set_ylabel("z_2")
ax.set_title("Vectorfield")
plt.tight_layout()

plt.show()
