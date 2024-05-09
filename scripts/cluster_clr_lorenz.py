import sys
sys.path.append('../')
from src import RandomFeatureConceptorRNN
import torch
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

# batch condition
steps = int(sys.argv[-2])
rep = int(sys.argv[-1])

# general
dtype = torch.float64
device = "cpu"
state_vars = ["x", "y", "z"]
lag = 1
noise_lvl = 1.0

# lorenz equation parameters
s = 10.0
r = 28.0
b = 8/3
dt = 0.01
init_steps = 1000

# reservoir parameters
N = 200
n_in = len(state_vars)
k = 10
sr = 0.99
bias_scale = 0.01
in_scale = 0.01
out_scale = 1.0
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
test_steps = 50000
loading_steps = int(0.5*steps)
backprop_steps = 5000
lam = 0.002
lr = 0.01
alpha = 4.0
betas = (0.9, 0.999)
tychinov = 1e-3
epsilon = 1e-8

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
        y = W_r @ rnn.forward_c_adapt(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))

        # calculate loss
        loss += loss_func(y, targets[step])

        # make update
        if (step + 1) % backprop_steps == 0:
            optim.zero_grad()
            W_r_tmp = torch.abs(rnn.W_z)
            for j in range(k):
                W_r_tmp[j, :] *= rnn.C[j]
            loss += epsilon*torch.sum(torch.abs(rnn.W) @ W_r_tmp)
            loss /= backprop_steps
            loss.backward()
            current_loss = loss.item()
            optim.step()
            loss = torch.zeros((1,))
            rnn.detach()

# store conceptor
rnn.store_conceptor("lorenz")
rnn.detach()

# harvest states
y_col = []
for step in range(loading_steps):
    rnn.forward_c(inputs[step] + noise_lvl*torch.randn((n_in,), device=device, dtype=dtype))
    y_col.append(rnn.y)
y_col = torch.stack(y_col, dim=0)

# train readout
W_r, epsilon = rnn.train_readout(y_col.T, targets[:loading_steps].T, tychinov)
epsilon = float(torch.mean(epsilon).cpu().detach().numpy())
print(f"Readout training error: {epsilon}")

# inspect conceptor
c = rnn.C.cpu().detach().numpy()
k_star = np.sum(c)
print(f"k_star: {k_star}")

# generate predictions
with torch.no_grad():
    predictions = []
    y = W_r @ rnn.y
    for step in range(test_steps):
        y = W_r @ rnn.forward_c(y)
        predictions.append(y.cpu().detach().numpy())
predictions = np.asarray(predictions)

# save results
results = {"targets": targets[loading_steps:loading_steps+test_steps], "predictions": predictions, "c": c,
           "config": {"N": N, "alpha": alpha, "sr": sr, "bias": bias_scale, "in": in_scale, "p": density, "k": k},
           "condition": {"steps": steps, "repetition": rep},
           "training_error": epsilon}
pickle.dump(results, open(f"../results/clr_lorenz/n{steps}_{rep}.pkl", "wb"))
