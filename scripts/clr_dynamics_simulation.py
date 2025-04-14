import sys
sys.path.append('../')
import torch
import numpy as np
from src.rnn import LowRankCRNN
from src.functions import init_weights, init_dendrites
import pickle

# parameter definition
######################

# general
dtype = torch.float64
device = "cpu"
file = "/home/richard/data/clr_dynamics.pkl"

# task parameters
steps = 500
init_steps = 50
epsilon = 1e-4

# rnn parameters
k = 200
n_in = 1
n_dendrites = 10
density = 0.5
N = int(k * n_dendrites)

# sweep parameters
in_scales = [0.1, 0.2]
Delta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
sigma = np.arange(start=0.0, stop=3.1, step=0.1)
n_reps = 20
n_trials = len(in_scales)*len(Delta)*len(sigma)*n_reps

# prepare results
results = {"in_scale": [], "Delta": [], "sigma": [], "trial": [], "z_init": [], "z_perturbed": [], "z_unperturbed": []}

# simulations
#############

with torch.no_grad():
    n = 0
    for in_scale in in_scales:
        for Delta_tmp in Delta:
            for sigma_tmp in sigma:
                for trial in range(n_reps):

                    # initialize rnn matrices
                    bias = torch.tensor(Delta_tmp * np.random.randn(k), device=device, dtype=dtype)
                    W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
                    L = init_weights(N, k, density)
                    W, R = init_dendrites(k, n_dendrites)

                    # model initialization
                    rnn = LowRankCRNN(torch.tensor(W, dtype=dtype, device=device),
                                      torch.tensor(L, dtype=dtype, device=device),
                                      torch.tensor(R, device=device, dtype=dtype), W_in, bias, g="ReLU")
                    rnn.C_z *= sigma_tmp

                    # input definition
                    inp = torch.randn((steps, n_in), device=device, dtype=dtype)

                    # get initial state
                    z0s = []
                    for step in range(init_steps):
                        x = torch.randn(n_in, dtype=dtype, device=device)
                        rnn.forward(x)
                    perturbed_state = [v[:] + epsilon*torch.randn(v.shape[0]) for v in rnn.state_vars]

                    # model simulation I
                    z1s = []
                    for step in range(steps):
                        z1s.append(rnn.z)
                        rnn.forward(inp[step])

                    # model simulation II
                    rnn.set_state(perturbed_state)
                    z2s = []
                    for step in range(steps):
                        z2s.append(rnn.z)
                        rnn.forward(inp[step])

                    # save results
                    results["in_scale"].append(in_scale)
                    results["Delta"].append(Delta_tmp)
                    results["sigma"].append(sigma_tmp)
                    results["trial"].append(trial)
                    results["z_init"].append(np.asarray(z0s))
                    results["z_unperturbed"].append(np.asarray(z1s))
                    results["z_perturbed"].append(np.asarray(z2s))

                    # report progress
                    n += 1
                    print(f"Finished {n} out of {n_trials} trials.")

# save results to file
######################

pickle.dump(results, open(file, "wb"))
