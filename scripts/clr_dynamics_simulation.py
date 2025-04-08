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
file = "/home/richard-gast/Documents/data/clr_dynamics.pkl"

# task parameters
steps = 100
init_steps = 20
init_noise = 1e-1
epsilon = 1e-2

# rnn parameters
n_in = 2
n_dendrites = 10
density = 0.5
in_scale = 1.0

# sweep parameters
k = [10, 100]
lam = [0.0, 0.1, 0.2]
Delta = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
sigma = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
n_reps = 10
n_trials = len(k)*len(lam)*len(Delta)*len(sigma)*n_reps*2

# prepare results
results = {"k": [], "lambda": [], "Delta": [], "sigma": [], "z": [], "z_p": [], "trial": [], "input": []}

# simulations
#############

with torch.no_grad():
    n = 0
    for k_tmp in k:
        for lam_tmp in lam:
            for Delta_tmp in Delta:
                for sigma_tmp in sigma:
                    for idx in range(n_in):
                        for trial in range(n_reps):

                            # initialize rnn matrices
                            N = int(k_tmp * n_dendrites)
                            bias = torch.tensor(Delta_tmp * np.random.randn(k_tmp), device=device, dtype=dtype)
                            W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
                            L = init_weights(N, k_tmp, density)
                            W, R = init_dendrites(k_tmp, n_dendrites)

                            # model initialization
                            rnn = LowRankCRNN(torch.tensor(W*lam_tmp, dtype=dtype, device=device),
                                              torch.tensor(L*(1-lam_tmp), dtype=dtype, device=device),
                                              torch.tensor(R, device=device, dtype=dtype), W_in, bias, g="ReLU")
                            rnn.C_z *= sigma_tmp

                            # input definition
                            inp = torch.zeros((steps, n_in), device=device, dtype=dtype)
                            inp[0, idx] = 1.0

                            # get initial state
                            for step in range(init_steps):
                                rnn.forward(init_noise * torch.randn(n_in, dtype=dtype, device=device))
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
                            results["k"].append(k_tmp)
                            results["lambda"].append(lam_tmp)
                            results["Delta"].append(Delta_tmp)
                            results["sigma"].append(sigma_tmp)
                            results["input"].append(idx)
                            results["trial"].append(trial)
                            results["z"].append(np.asarray(z1s))
                            results["z_p"].append(np.asarray(z2s))

                            # report progress
                            n += 1
                            print(f"Finished {n} out of {n_trials} trials.")

# save results to file
######################

pickle.dump(results, open(file, "wb"))
