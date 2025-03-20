from src import ConceptorLowRankRNN
from src.functions import init_weights
import torch
import matplotlib.pyplot as plt
import numpy as np


# function definitions
######################

def get_inp(f1: float, f2: float, trial_dur: int, steps: int, noise: float, dt: float) -> tuple:

    # create sines
    time = np.linspace(0.0, steps*dt, steps)
    s1 = np.sin(2.0*np.pi*f1*time)
    s2 = np.sin(2.0*np.pi*f2*time)

    # create switching signal and targets
    switch = np.zeros_like(s1)
    step = 0
    targets = np.zeros((steps, 1))
    while step < steps:
        if np.random.randn() > 0:
            switch[step:step+trial_dur] = 1.0
            targets[step:step+trial_dur, 0] = s1[step:step+trial_dur]
        else:
            targets[step:step+trial_dur, 0] = s2[step:step+trial_dur]
        step += trial_dur

    # add noise to input signals
    inputs = np.asarray([s1, s2, switch]).T + noise * np.random.randn(steps, 3)

    return torch.tensor(inputs, device=device, dtype=dtype), torch.tensor(targets, device=device, dtype=dtype)


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 2000

# input parameters
n_in = 3
f1 = 0.5
f2 = 3.0
dt = 0.01
trial_dur = 300
noise = 0.1

# reservoir parameters
N = 100
k = 50
sr = 1.1
bias_scale = 0.01
in_scale = 0.1
density = 0.2

# training parameters
train_steps = 600000
loading_steps = 100000
test_steps = 18000
init_steps = 1000
lam = 0.002
alphas = (12.0, 1e-3)

# matrix initialization
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)

# generate inputs and targets
#############################

# get training data
x_train, y_train = get_inp(f1, f2, trial_dur, train_steps, noise, dt)

# get loading data
x_load, y_load = get_inp(f1, f2, trial_dur, loading_steps, noise, dt)

# get test data
x_test, y_test = get_inp(f1, f2, trial_dur, test_steps, noise, dt)

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = ConceptorLowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                          torch.tensor(W_z, device=device, dtype=dtype), lam, alphas[0])
rnn.init_new_conceptor(init_value="random")

# initial wash-out period
avg_input = torch.mean(x_train, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward_c(avg_input)

# train the conceptor
with torch.no_grad():
    for step in range(train_steps):
        x = rnn.forward_c_adapt(x_train[step])

# harvest states
y_col = []
for step in range(loading_steps):
    rnn.forward_c(x_load[step])
    y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

# train readout
W_r, epsilon = rnn.train_readout(y_col.T, y_load.T, alphas[1])
print(f"Readout training error: {float(torch.mean(epsilon).cpu().detach().numpy())}")

# retrieve network connectivity
c = rnn.C.cpu().detach().numpy().squeeze()
W = (rnn.L @ (torch.diag(rnn.C) @ rnn.L)).cpu().detach().numpy()
W_abs = np.sum((torch.abs(rnn.L) @ torch.abs(torch.diag(rnn.C) @ rnn.L)).cpu().detach().numpy())
print(f"Conceptor: {np.sum(c)}")

# generate predictions
with torch.no_grad():
    predictions = []
    for step in range(test_steps):
        y = W_r @ rnn.forward_c(x_test[step])
        predictions.append(y.cpu().detach().numpy())
predictions = np.asarray(predictions)
targets = y_test.cpu().detach().numpy()

# plotting
##########

# dynamics
_, ax = plt.subplots(figsize=(12, 6))
ax.plot(targets[:plot_steps], color="royalblue", label="target")
ax.plot(predictions[:plot_steps], color="darkorange", label="prediction")
ax.set_xlabel("steps")
plt.tight_layout()

# trained weights
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="equal", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
fig.suptitle(f"Absolute weights: {np.round(W_abs, decimals=1)}")
plt.tight_layout()
plt.show()
