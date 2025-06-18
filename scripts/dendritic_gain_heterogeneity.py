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
file = "/home/richard/data/dendritic_gain_dynamics.pkl"

# task parameters
steps = 500
epsilon = 1e-5

# rnn parameters
k = 200
n_in = 1
n_dendrites = 10
density = 0.5
in_scale = 0.1
N = int(k * n_dendrites)

# sweep parameters
sigmas = np.arange(start=0.0, stop=2.1, step=0.1)
lambdas = [0.0, 0.5, 1.0, 1.5, 2.0]
Deltas = [0.0, 0.25, 0.5, 1.0]
n_reps = 20
n_trials = len(Deltas) * len(lambdas) * len(sigmas) * n_reps

# prepare results
results = {"Delta": [], "lambda": [], "sigma": [], "trial": [], "x": [], "z_noinp": [], "z_inp": [], "z_inp_p": []}

# simulations
#############

with torch.no_grad():
    n = 0
    for trial in range(n_reps):

        # initialize rnn matrices
        bias = torch.zeros(k, device=device, dtype=dtype)
        W_in = torch.tensor(np.random.randn(N, n_in), device=device, dtype=dtype)
        L = torch.tensor(init_weights(N, k, density), device=device, dtype=dtype)
        W, R = init_dendrites(k, n_dendrites)
        R = torch.tensor(R, device=device, dtype=dtype)

        # input definition
        inp = torch.randn((steps, n_in), device=device, dtype=dtype)

        # get initial and perturbed state
        successful = False
        while not successful:
            init_state = [2*torch.rand(N, device=device, dtype=dtype) - 1.0,
                          2.0*torch.rand(k, device=device, dtype=dtype)]
            perturbed_state = [v[:] + epsilon * torch.randn(v.shape[0]) for v in init_state]
            diffs = [torch.sum((v - v_p) ** 2) for v, v_p in zip(init_state, perturbed_state)]
            if all([d.item() > 0 for d in diffs]):
                successful = True
        for Delta in Deltas:
            for lam in lambdas:
                for sigma in sigmas:

                    # model initialization
                    W_tmp = torch.tensor(W*lam, dtype=dtype, device=device)
                    dendritic_gains = np.random.uniform(low=1.0-Delta, high=1.0+Delta, size=N)
                    rnn = LowRankCRNN(W_tmp, L*sigma, R, W_in * in_scale, bias, g="ReLU")
                    rnn.C_y = torch.tensor(dendritic_gains, dtype=dtype, device=device)

                    # simulation a - zero input
                    rnn.set_state(init_state)
                    z0s = []
                    x = torch.zeros(n_in, dtype=dtype, device=device)
                    for step in range(steps):
                        z0s.append(rnn.z)
                        rnn.forward(x)

                    # simulation b - random input
                    rnn.set_state(init_state)
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
                    results["Delta"].append(Delta)
                    results["lambda"].append(lam)
                    results["sigma"].append(sigma)
                    results["trial"].append(trial)
                    results["x"].append(inp)
                    results["z_noinp"].append(np.asarray(z0s))
                    results["z_inp"].append(np.asarray(z1s))
                    results["z_inp_p"].append(np.asarray(z2s))

                    # report progress
                    n += 1
                    print(f"Finished {n} out of {n_trials} trials.")

# save results to file
######################

pickle.dump(results, open(file, "wb"))
