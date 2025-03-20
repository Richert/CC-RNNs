from src.rnn import ConceptorLowRankOnlyRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights, init_dendrites


# function definitions
######################

def stuart_landau(y: torch.Tensor, x: torch.Tensor, mu: float, a: float, b: float, omega: float = 10.0) -> torch.Tensor:
    y1_dot = (x + mu)*y[0] - a*torch.abs(y[0]**3)
    y2_dot = omega + b*y[0]
    return torch.stack([y1_dot, y2_dot], dim=0)

# parameter definition
######################

# general
dtype = torch.float64
device = "cuda:0"
plot_steps = 2000
state_vars = ["r", "theta"]

# task parameters
omegas = [4, 9]
tau = 10
mu = -1.0
a = 20.0
b = 30.0
dt = 0.01
steps = 2000
trials = 100
init_steps = 1000

# rnn parameters
n_in = len(state_vars)
k = 100
n_dendrites = 3
N = int(k*n_dendrites)
sr = 0.99
bias_scale = 0.01
in_scale = 1.0
out_scale = 0.1
density = 0.5

# initialize rnn matrices
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
L = init_weights(N, k, density)
R = init_dendrites(k, n_dendrites)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(L, R))))
L *= np.sqrt(sr) / np.sqrt(sr_comb)
R *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_in, N), device=device, dtype=dtype)

# fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
# ax = axes[0]
# ax.imshow(L, aspect="auto", interpolation="none")
# ax.set_xlabel("from: neurons")
# ax.set_ylabel("to: dendrites")
# ax = axes[1]
# ax.imshow(R, aspect="auto", interpolation="none")
# ax.set_xlabel("from: dendrites")
# ax.set_ylabel("to: neurons")
# plt.tight_layout()
# plt.show()

# training parameters
backprop_steps = 1000
train_trials = int(0.9*trials)
test_trials = steps - train_trials
lr = 0.05
lam = 0.002
alpha = 5.0
betas = (0.9, 0.999)

# conceptors
conceptors = {}
for i, omega in enumerate(omegas):
    conceptors[omega] = torch.rand((k,), device=device, dtype=dtype)

# generate input and targets
############################

# generate inputs
y0 = torch.zeros((2,), device=device, dtype=dtype)
inputs_col = {}
for omega in omegas:
    inputs_col[omega] = []
    for trial in range(trials):
        y_col = []
        y = y0.detach()
        x = torch.randn(steps + tau, dtype=dtype, device=device)
        for step in range(steps + tau):
            y = y + dt * stuart_landau(y, x=x[step], omega=omega, mu=mu, a=a, b=b)
            y_col.append(y.detach().cpu().numpy())
        inputs_col[omega].append(y_col)

# generate targets
targets_col = {}
for omega in omegas:
    inp = np.asarray(inputs_col[omega])
    targets_col[omega] = inp[:, tau:, :]
    inputs_col[omega] = inp[:, :-tau, :]

# train LR-RNN weights
######################

# initialize RFC
rnn = ConceptorLowRankOnlyRNN(W_in, bias, torch.tensor(L, dtype=dtype, device=device),
                              torch.tensor(R, device=device, dtype=dtype), alpha=alpha, lam=lam)
rnn.conceptors.update(conceptors)
rnn.free_param("L")
rnn.free_param("W_in")

# initial wash-out period
avg_input = torch.zeros((n_in,), device=device, dtype=dtype)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward_c(avg_input)
init_state = rnn.y[:]

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# training
loss = torch.zeros((1,))
for trial in range(train_trials):

    for omega in omegas:

        # initial condition
        rnn.activate_conceptor(omega)
        rnn.set_state(init_state)

        # get inputs and targets
        inputs = torch.tensor(inputs_col[omega][trial], device=device, dtype=dtype)
        targets = torch.tensor(targets_col[omega][trial], device=device, dtype=dtype)

        # training the RNN weights
        current_loss = 0.0
        with torch.enable_grad():

            y = y0.detach()
            for step in range(steps):

                # get RNN readout
                x = W_r @ rnn.forward_c(inputs[step])
                y = y + dt * stuart_landau(y, x=x, omega=omega, mu=mu, a=a, b=b)

                # calculate loss
                loss += loss_func(y, targets[step])

    # make update
    optim.zero_grad()
    loss.backward()
    current_loss = loss.item()
    optim.step()
    loss = torch.zeros((1,))
    rnn.detach()
    print(f"Training trial {trial + 1}: MSE = {current_loss}")

# generate predictions
######################

predictions_col = {}
loss = torch.zeros((1,))
for omega in omegas:

    predictions_col[omega] = []

    for trial in range(train_trials, trials):

        # initial condition
        rnn.activate_conceptor(omega)
        rnn.set_state(init_state)

        # get inputs and targets
        inputs = torch.tensor(inputs_col[omega][trial], device=device, dtype=dtype)
        targets = torch.tensor(targets_col[omega][trial], device=device, dtype=dtype)

        # training the RNN weights
        current_loss = 0.0
        with torch.no_grad():

            y = y0.detach()
            predictions = []
            for step in range(steps):

                # get RNN readout
                x = W_r @ rnn.forward_c(inputs[step])
                y = y + dt * stuart_landau(y, x=x, omega=omega, mu=mu, a=a, b=b)
                predictions.append(y.detach().cpu().numpy())

        predictions_col[omega].append(predictions)
    predictions_col[omega] = np.asarray(predictions_col[omega])

# plotting
##########

# prediction figure
n_examples = 2
n_rows = len(omegas)
fig, axes = plt.subplots(ncols=n_in, nrows=n_rows, figsize=(12, 6))
for i in range(n_rows):
    for j in range(n_in):
        ax = axes[i, j]
        for example in range(n_examples):
            trial = np.random.randint(low=0, high=test_trials)
            ax.plot(targets_col[omegas[i]][train_trials+trial, :plot_steps, j], label="target", linestyle="dashed")
            ax.plot(predictions_col[omegas[i]][trial, :plot_steps, j], label="prediction", linestyle="solid")
        ax.set_ylabel(state_vars[j])
        if i == n_rows-1:
            ax.set_xlabel("steps")
            ax.legend()

plt.tight_layout()
fig.suptitle("Model Predictions")
