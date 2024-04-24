from src import RandomFeatureConceptorRNN
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
plot_steps = 4000
state_vars = ["x", "y", "z"]
lag = 1

# lorenz equation parameters
s = 10.0
r = 28.0
b = 8/3
dt = 0.01
steps = 400000
init_steps = 1000

# reservoir parameters
N = 200
n_in = len(state_vars)
k = 10
sr = 0.99
bias_scale = 0.01
in_scale = 0.01
out_scale = 0.5
density = 0.1

# matrix initialization
W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
bias = torch.tensor(bias_scale * np.random.randn(N), device=device, dtype=dtype)
W = init_weights(N, k, density)
W_z = init_weights(k, N, density)
sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
W *= np.sqrt(sr) / np.sqrt(sr_comb)
W_z *= np.sqrt(sr) / np.sqrt(sr_comb)
W_r = torch.tensor(out_scale * np.random.randn(n_in, N), device=device, dtype=dtype)

# training parameters
test_steps = 4500
loading_steps = int(0.5*steps)
backprop_steps = 500
lam = 0.002
lr = 0.01
alpha = 4.0
betas = (0.9, 0.999)
tychinov = 1e-3
epsilon = 1e-6

# generate inputs and targets
#############################

# simulation
y = np.asarray([0.1, 0.9, 0.5])
y_col = []
for step in range(steps):
    y = y + dt * lorenz(y[0], y[1], y[2], s=s, r=r, b=b)
    y_col.append(y)
y_col = np.asarray(y_col)

# get inputs and targets
inputs = torch.tensor(y_col[:-lag], device=device, dtype=dtype)
targets = torch.tensor(y_col[lag:], device=device, dtype=dtype)

# train RFC-RNN to predict next time step of Lorenz attractor
#############################################################

# initialize RFC-RNN
rnn = RandomFeatureConceptorRNN(torch.tensor(W, dtype=dtype, device=device), W_in, bias,
                                torch.tensor(W_z, device=device, dtype=dtype), lam, alpha)
rnn.free_param("W")
rnn.free_param("W_z")
rnn.init_new_conceptor(init_value="random")

# initial wash-out period
avg_input = torch.mean(inputs, dim=0)
with torch.no_grad():
    for step in range(init_steps):
        rnn.forward_c(avg_input)

# set up loss function
loss_func = torch.nn.MSELoss()

# set up optimizer
optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)

# train the RNN weights and the conceptor simultaneously
current_loss = 0.0
with torch.enable_grad():

    loss = torch.zeros((1,))
    for step in range(steps-lag):

        # get RNN readout
        y = W_r @ rnn.forward_c_adapt(inputs[step])

        # calculate loss
        loss += loss_func(y, targets[step])

        # make update
        if (step + 1) % backprop_steps == 0:
            optim.zero_grad()
            W_r_tmp = torch.abs(rnn.W_z)
            for j in range(k):
                W_r_tmp[j, :] *= rnn.C[j]
            loss += epsilon*torch.sum(torch.abs(rnn.W) @ W_r_tmp)
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()
            print(f"Training phase loss: {current_loss}")

# store conceptor
rnn.store_conceptor("lorenz")
rnn.detach()

# harvest states
y_col = []
for step in range(loading_steps):
    rnn.forward_c(inputs[step])
    y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

# load input into RNN weights
D, epsilon = rnn.load_input(y_col.T, inputs[1:loading_steps+1].T, tychinov)
print(f"Input loading error: {float(torch.mean(epsilon).cpu().detach().numpy())}")

# train readout
W_r, epsilon2 = rnn.train_readout(y_col.T, targets[:loading_steps].T, tychinov)
print(f"Readout training error: {float(torch.mean(epsilon2).cpu().detach().numpy())}")

# finalize conceptors
# with torch.no_grad():
#     rnn.y, rnn.z = y0, z0
#     for step in range(loading_steps):
#         rnn.forward_c_a_adapt()
c = rnn.C.cpu().detach().numpy()
print(f"Conceptor: {np.sum(c)}")

# generate predictions
with torch.no_grad():
    predictions = []
    for step in range(test_steps):
        y = rnn.forward_c_a()
        y = W_r @ y
        predictions.append(y.cpu().detach().numpy())
predictions = np.asarray(predictions)

# save results
results = {"targets": targets, "predictions": predictions,
           "config": {"N": N, "k": k, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "lam": lam,
                      "alpha": alpha, "lag": lag}}
pickle.dump(results, open("../results/lr_rfc_lorenz.pkl", "wb"))

# plotting
##########

fig, axes = plt.subplots(nrows=n_in, figsize=(12, 6))

for i, ax in enumerate(axes):

    ax.plot(targets[:plot_steps, i], color="royalblue", label="target")
    ax.plot(predictions[:plot_steps, i], color="darkorange", label="prediction")
    ax.set_ylabel(state_vars[i])
    if i == n_in-1:
        ax.set_xlabel("steps")
        ax.legend()

plt.tight_layout()
plt.show()
