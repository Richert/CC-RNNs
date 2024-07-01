from src import ConceptorLowRankRNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.functions import init_weights
import pickle


# function definitions
######################

def lorenz(x: float, y: float, z: float, s: float = 10.0, r: float = 28.0, b: float = 2.667) -> np.ndarray:
    """
    Parameters
    ----------
    x, y, z: float
        State variables.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : np.ndarray
       Vectorfield of the Lorenz equations.
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.asarray([x_dot, y_dot, z_dot])


# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
plot_steps = 8000

# input parameters
in_vars = ["x", "y"]
n_epochs = 2050
train_epochs = 2000
epoch_steps = 200
signal_scale = 0.5
noise_scale = 0.1
steps = int(n_epochs*epoch_steps)

# reservoir parameters
N = 200
n_in = len(in_vars)
k = 512
sr = 1.05
bias_scale = 0.01
in_scale = 0.01
density = 0.1

# matrix initialization
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)

# training parameters
loss_start = int(0.5*epoch_steps)
loss_epochs = 15
lam = 0.002
lr = 0.05
alpha = 15.0
betas = (0.9, 0.999)

# generate inputs and targets
#############################

inputs, targets = [], []
for n in range(n_epochs):

    mus = signal_scale * torch.randn((n_in,), device=device, dtype=dtype)
    inp = torch.zeros((epoch_steps, n_in), device=device, dtype=dtype)
    for i, mu in enumerate(mus):
        inp[:, i] += mu + noise_scale * torch.randn((epoch_steps,), device=device, dtype=dtype)
    targ = torch.zeros_like(inp)
    targ[:, torch.argmax(mus)] = 1.0
    inputs.append(inp)
    targets.append(targ)

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = ConceptorLowRankRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                          torch.tensor(W_z, device=device, dtype=dtype), lam, alpha)
rnn.init_new_conceptor(init_value="random")
readout = torch.nn.Linear(in_features=N, out_features=n_in, bias=False, device=device, dtype=dtype)

# initial wash-out period
avg_input = torch.mean(inputs[0], dim=0)
with torch.no_grad():
    for step in range(epoch_steps):
        rnn.forward_c(avg_input)

# train the conceptor
with torch.no_grad():
    for epoch in range(train_epochs):
        for step in range(epoch_steps):
            rnn.forward_c_adapt(inputs[epoch][step])

# train readout
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(readout.parameters(), lr=lr, betas=betas)
current_loss = 0.0
with torch.enable_grad():

    loss = torch.zeros((1,))
    for epoch in range(train_epochs):
        for step in range(epoch_steps):

            # get RNN output
            y = readout.forward(rnn.forward(inputs[epoch][step]))

            # calculate loss
            if step > loss_start:
                loss += loss_func(y, targets[epoch][step])

        # make update
        if (epoch + 1) % loss_epochs == 0:
            optim.zero_grad()
            loss.backward()
            current_loss = loss.item()
            optim.step()
            rnn.detach()
            loss = torch.zeros((1,))
            print(f"Readout training loss (epoch {epoch}): {current_loss}")

# inspect conceptor
c = rnn.C.cpu().detach().numpy()
print(f"Conceptor: {np.sum(c)}")

# generate predictions
with torch.no_grad():
    evidences, test_targets, test_inputs = [], [], []
    for epoch in range(n_epochs-train_epochs):
        for step in range(epoch_steps):
            inp = inputs[train_epochs+epoch][step]
            y = readout.forward(rnn.forward(inp))
            evidences.append(y.cpu().detach().numpy())
            test_targets.append(targets[train_epochs+epoch][step].cpu().detach().numpy())
            test_inputs.append(inp.cpu().detach().numpy())
evidences = np.asarray(evidences)
test_targets = np.asarray(test_targets)
test_inputs = np.asarray(test_inputs)

# save results
# results = {"targets": targets, "predictions": predictions,
#            "config": {"N": N, "k": k, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "lam": lam,
#                       "alpha": alpha, "lag": lag}}
# pickle.dump(results, open("../results/rfc_lorenz.pkl", "wb"))

# plotting
##########

fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))

for i, ax in enumerate(axes):

    # ax.plot(test_inputs[:plot_steps, i], color="black", label="input", alpha=0.5)
    ax.plot(test_targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(evidences[:plot_steps, i] / np.max(evidences[:plot_steps, i]), color="darkorange", label="evidence")
    ax.plot(evidences[:plot_steps, i] > evidences[:plot_steps, 1-i], color="darkgreen", label="prediction")
    ax.set_ylabel(in_vars[i])
    if i == n_in-1:
        ax.set_xlabel("steps")
        ax.legend()

plt.tight_layout()
plt.show()
