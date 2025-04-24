
import sys
sys.path.append("../")
from src.rnn import LowRankCRNN
from src.functions import init_weights, init_dendrites
import torch
import numpy as np
import pickle

# parameter definition
######################

# general
n_conditions = 3
dtype = torch.float64
device = "cuda:0"
state_vars = ["y"]
path = "/home/richard"
load_file = f"{path}/data/vdp_{n_conditions}freqs.pkl"
save_file = f"{path}/results/clr_rhythmic_{n_conditions}freqs_cfit_noweights.pkl"

# load inputs and targets
data = pickle.load(open(load_file, "rb"))
inputs = data["inputs"]
targets = data["targets"]
conditions = data["trial_conditions"]
unique_conditions = np.unique(conditions)

# task parameters
steps = inputs[0].shape[0]
init_steps = 100
noise_lvl = 0.01

# add noise to input
inputs = [inp[:, :1] + noise_lvl * np.random.randn(inp.shape[0], inp.shape[1]-1) for inp in inputs]

# rnn parameters
k = 100
n_dendrites = 10
n_in = inputs[0].shape[-1]
n_out = targets[0].shape[-1]
density = 0.5
in_scale = 0.1
out_scale = 0.2
N = int(k * n_dendrites)

# training parameters
trials = len(conditions)
train_trials = int(0.9 * trials)
test_trials = trials - train_trials
augmentation = 1.0
lr = 1e-2
betas = (0.9, 0.999)
batch_size = 50
gradient_cutoff = 1e4
truncation_steps = 100
epsilon = 0.1
lam = 1e-3
alpha = 10.0
batches = int(augmentation * train_trials / batch_size)

# sweep parameters
Delta = [0.1, 0.4]
sigma = np.arange(start=0.2, stop=1.7, step=0.2)
n_reps = 10
n_trials = len(Delta)*len(sigma)*n_reps

# prepare results
results = {"Delta": [], "sigma": [], "trial": [], "train_epochs": [], "train_loss": [], "test_loss": []}

# model training
################

n = 0
for Delta_tmp in Delta:
    for sigma_tmp in sigma:
        for rep in range(n_reps):

            print(f"Starting run {n+1} out of {n_trials} training runs (Delta = {Delta_tmp}, sigma = {sigma_tmp}, rep = {rep})")

            # initialize rnn matrices
            bias = torch.tensor(Delta_tmp * np.random.randn(k), device=device, dtype=dtype)
            W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
            L = init_weights(N, k, density)
            W, R = init_dendrites(k, n_dendrites)
            W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

            # model initialization
            rnn = LowRankCRNN(torch.tensor(W*0.0, dtype=dtype, device=device),
                              torch.tensor(L*sigma_tmp, dtype=dtype, device=device),
                              torch.tensor(R, device=device, dtype=dtype),
                              W_in, bias, g="ReLU", alpha=alpha, lam=lam)
            rnn.free_param("W_in")

            # initialize controllers
            conceptors = []
            for c in unique_conditions:
                rnn.init_new_y_controller(init_value="random")
                rnn.C_y.requires_grad = True
                conceptors.append(rnn.C_y)
                rnn.store_y_controller(c)

            # set up loss function
            loss_func = torch.nn.MSELoss()

            # set up optimizer
            optim = torch.optim.Adam(list(rnn.parameters()) + conceptors + [W_r], lr=lr, betas=betas)
            rnn.clip(gradient_cutoff)

            # training
            train_loss = 0.0
            loss_col = []
            with torch.enable_grad():
                for batch in range(batches):

                    loss = torch.zeros((1,), device=device, dtype=dtype)

                    for trial in np.random.choice(train_trials, size=(batch_size,), replace=False):

                        # get input and target timeseries
                        inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
                        target = torch.tensor(targets[trial], device=device, dtype=dtype)

                        # get initial state
                        for step in range(init_steps):
                            x = torch.randn(n_in, dtype=dtype, device=device)
                            rnn.forward(x)
                        rnn.detach()
                        rnn.activate_y_controller(conditions[trial])

                        # collect loss
                        y_col = []
                        for step in range(steps):
                            z = rnn.forward(inp[step])
                            y = W_r @ z
                            if step % truncation_steps == truncation_steps - 1:
                                rnn.detach()
                            y_col.append(y)

                        # calculate loss
                        y_col = torch.stack(y_col, dim=0)
                        loss += loss_func(y_col, target)

                        # store controller
                        rnn.store_y_controller(conditions[trial])

                    # make update
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    # store and print loss
                    train_loss = loss.item()
                    loss_col.append(train_loss)
                    if train_loss < epsilon:
                        break

            # generate predictions
            test_loss = []
            with torch.no_grad():
                for trial in range(train_trials, trials):

                    # get input and target timeseries
                    inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
                    target = torch.tensor(targets[trial], device=device, dtype=dtype)

                    # get initial state
                    for step in range(init_steps):
                        x = torch.randn(n_in, dtype=dtype, device=device)
                        rnn.forward(x)
                    rnn.detach()
                    rnn.activate_y_controller(conditions[trial])

                    # make prediction
                    y_col = []
                    for step in range(steps):
                        z = rnn.forward(inp[step])
                        y = W_r @ z
                        y_col.append(y)

                    # calculate loss
                    loss = loss_func(torch.stack(y_col, dim=0), target)
                    test_loss.append(loss.item())

            # save results
            results["Delta"].append(Delta_tmp)
            results["sigma"].append(sigma_tmp)
            results["trial"].append(rep)
            results["train_epochs"].append(batch)
            results["train_loss"].append(loss_col)
            results["test_loss"].append(np.sum(test_loss))

            # report progress
            n += 1
            print(f"Finished after {batch + 1} training epochs. Final loss: {loss_col[-1]}.")

        # save results
        pickle.dump(results, open(save_file, "wb"))
