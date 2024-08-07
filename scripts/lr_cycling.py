from src import LowRankOnlyRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights
from scripts.task_functions import cycling


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 1000

# input parameters
amp = 5.0
freq = 6.0
dt = 0.01
n_in = 2
trial_dur = 200
min_cycling_dur = 30
inp_dur = 5
inp_damping = 1.0
padding = int(0.2*trial_dur)
inp_noise = 0.1
avg_input = torch.zeros(size=(n_in,), device=device, dtype=dtype)

# reservoir parameters
N = 200
n_out = 1
k = 4
sr = 1.0
bias_scale = 0.01
in_scale = 0.2
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
n_train = 10000
n_test = 1000
init_steps = 1000
batch_size = 10
lr = 0.0005
betas = (0.9, 0.999)
alphas = (1.0, 1e-5)

# generate inputs and targets
#############################

# get training data
x_train, y_train = cycling(amp, freq, trial_dur, min_cycling_dur, inp_dur, inp_damping, n_train, inp_noise, dt,
                           device=device, dtype=dtype)

# get test data
x_test, y_test = cycling(amp, freq, trial_dur, min_cycling_dur, inp_dur, inp_damping, n_test, inp_noise, dt,
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
    rnn.forward(avg_input)
y0 = rnn.y.detach()
z0 = rnn.z.detach()

# training
current_loss = 100.0
min_loss = 0.01
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
            loss += alphas[1] * torch.sum(torch.abs(rnn.L) @ torch.abs(rnn.R))
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()
            loss_hist.append(current_loss)
            print(f"MSE loss after {trial+1} training trials: {current_loss}")

        if current_loss < min_loss:
            break

W = (rnn.L @ rnn.R).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(rnn.R)).cpu().detach().numpy())
z_col = np.asarray(z_col)

# testing
#########

# generate predictions
predictions, targets = [], []
z_test = []
inp_test = []
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
        trial_inp_dur = []
        for step in range(trial_dur):
            y = W_r @ rnn.forward(trial_inp[step])
            trial_predictions.append(y)
            targets.append(trial_targ[step])
            trial_z.append(rnn.z.detach().cpu().numpy())
            if trial_inp[step, 0] > 0.9 or trial_inp[step, 1] < -0.9:
                trial_inp_dur.append(1.0)
            else:
                trial_inp_dur.append(0.0)
        z_test.append(np.asarray(trial_z))
        inp_test.append(np.asarray(trial_inp_dur))
        trial_predictions = torch.stack(trial_predictions, dim=0)
        predictions.extend(trial_predictions.cpu().detach().numpy().tolist())

# calculate performance on test data
predictions = np.asarray(predictions)
targets = np.asarray(targets)

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
_, axes = plt.subplots(nrows=n_out, figsize=(12, 3*n_out), sharex=True)
for i in range(n_out):
    ax = axes[i] if n_out > 1 else axes
    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(f"y_{i+1}")
    if i == n_out-1:
        ax.set_xlabel("steps")
        ax.legend()
plt.tight_layout()

# trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Summed LR connectivity: {np.round(W_abs, decimals=1)}")

# loss history
_, ax = plt.subplots(figsize=(10, 3))
ax.plot(loss_hist)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.tight_layout()

# vector field
plot_trials = 5
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(coords[:, 0], coords[:, 1], vf[:, 0], vf[:, 1])
for _ in range(plot_trials):
    idx = np.random.randint(0, n_test)
    trial_z = z_test[idx]
    trial_inp = inp_test[idx]
    l = ax.plot(trial_z[trial_inp < 0.5, 0], trial_z[trial_inp < 0.5, 1], linestyle="dotted")
    ax.plot(trial_z[trial_inp > 0.5, 0], trial_z[trial_inp > 0.5, 1], linestyle="solid", c=l[0].get_color())
    ax.scatter(trial_z[0, 0], trial_z[0, 1], marker="o", s=50.0, c=l[0].get_color())
    ax.scatter(trial_z[-1, 0], trial_z[-1, 1], marker="x", s=50.0, c=l[0].get_color())
ax.set_xlabel("z_1")
ax.set_ylabel("z_2")
ax.set_title("Vectorfield")
plt.tight_layout()

plt.show()
