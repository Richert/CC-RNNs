import sys
sys.path.append("../")
from src.rnn import LowRankCRNN
import torch
import numpy as np
import pickle


def simulation(model: LowRankCRNN, inp: torch.Tensor, init_steps: int) -> list:

    # get initial state
    for step in range(init_steps):
        x = torch.randn(inp.shape[1], dtype=dtype, device=device)
        z = model.forward(x)
        y = W_r @ z

    # get dynamics for PF system
    z_col = []
    for step in range(inp.shape[0]):
        inp[step, :n_out] = y
        z = model.forward(inp[step])
        y = W_r @ z
        z_col.append(z.detach().cpu().numpy())

    return z_col

# parameter definition
######################

# general
dtype = torch.float64
device = "cuda:0"
state_vars = ["y"]
path = "/home/richard"
model_file = f"{path}/results/clr_bifurcations_zfit.pkl"
save_file = f"{path}/results/simulations_bifurcations_zfit.pkl"

# task parameters
steps = 1000
init_steps = 20
min_mu, max_mu = -0.5, 0.5

# load RNN parameters
params = pickle.load(open(model_file, "rb"))
k = len(params["bias"][0])
N = params["L"][0].shape[0]

# find relevant model fits
lam = 4e-4
idx = np.argwhere(np.asarray(params["lambda"]) == lam).squeeze()
n_fits = len(idx)

# prepare results
results = {"pf": [], "vdp": [], "pf_vdp": []}

# model training
################

with torch.no_grad():
    for i in idx:

        # initialize rnn matrices
        bias = torch.tensor(params["bias"][i], device=device, dtype=dtype)
        W_in = torch.tensor(params["W_in"][i], device=device, dtype=dtype)
        L = torch.tensor(params["L"][i], device=device, dtype=dtype)
        R = torch.tensor(params["R"][i], device=device, dtype=dtype)
        W_r = torch.tensor(params["W_out"][i], device=device, dtype=dtype)
        n_in, n_out = W_in.shape[1], W_r.shape[0]

        # model initialization
        rnn = LowRankCRNN(torch.empty(size=(N, N), dtype=dtype, device=device), L, R, W_in, bias,
                          g="ReLU")

        # add dendritic gain controllers
        conceptors = {}
        c_dim = {}
        for key, c in enumerate(params["conceptors"][i]):
            conceptors[key] = c
            c_dim[key] = np.sum(c ** 2)
            rnn.z_controllers[key] = torch.tensor(c, device=device, dtype=dtype)

        # generate model dynamics
        with torch.no_grad():

            # get PF dynamics
            inp = torch.zeros((steps, n_in), device=device, dtype=dtype)
            mus = torch.linspace(min_mu, max_mu, steps, device=device, dtype=dtype)
            inp[:, -1] = mus
            rnn.activate_z_controller(0)
            results["pf"].append(simulation(rnn, inp, init_steps=init_steps))

            # get VDP dynamics
            inp = torch.zeros((steps, n_in), device=device, dtype=dtype)
            inp[:, -1] = mus
            rnn.activate_z_controller(1)
            results["vdp"].append(simulation(rnn, inp, init_steps=init_steps))

            # get FP to VDP dynamics for mu = 0.2
            rnn.activate_z_controller(0)
            inp = torch.zeros((int(0.25*steps), n_in), device=device, dtype=dtype)
            inp[:, -1] = 0.2
            pf1 = simulation(rnn, inp.clone(), init_steps=1)
            rnn.activate_z_controller(1)
            vdp1 = simulation(rnn, inp.clone(), init_steps=1)
            rnn.activate_z_controller(0)
            pf2 = simulation(rnn, inp.clone(), init_steps=1)
            rnn.activate_z_controller(1)
            vdp2 = simulation(rnn, inp.clone(), init_steps=1)
            results["pf_vdp"].append(np.concatenate([pf1, vdp1, pf2, vdp2], axis=0))

# save results
pickle.dump(results, open(save_file, "wb"))
