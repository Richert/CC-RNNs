import sys
sys.path.append("../")
from src.rnn import LowRankCRNN
from src.functions import init_weights, init_dendrites
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

# parameter definition
######################

# general
dtype = torch.float64
device = "cuda:0"
state_vars = ["y"]
path = "/home/richard-gast/Documents"
in_file = f"{path}/data/bifurcations_2ds.pkl"
model_file = f"{path}/results/clr_bifurcations_zfit.pkl"
save_file = f"{path}/results/test_bifurcations_zfit.pkl"
visualize_results = True
plot_examples = 6

# load inputs and targets
data = pickle.load(open(in_file, "rb"))
inputs = data["inputs"]
targets = data["targets"]
conditions = data["trial_conditions"]
unique_conditions = np.unique(conditions, axis=0)
n_conditions = len(unique_conditions)

# task parameters
steps = inputs[0].shape[0]
init_steps = 20
auto_steps = 200
noise_lvl = 0.0

# load RNN parameters
params = pickle.load(open(model_file, "rb"))
n_in = inputs[0].shape[-1]
n_out = targets[0].shape[-1]
k = len(params["bias"][0])
N = params["L"][0].shape[0]

# training parameters
trials = len(conditions)
train_trials = int(0.9 * trials)
test_trials = trials - train_trials
batch_size = 20

# sweep parameters
alphas = [10.0, 12.5, 15.0]
n_reps = 10
n_trials = len(alphas)*n_reps

# prepare results
results = {"alpha": [], "trial": [], "train_loss": [], "test_loss": [], "z_dim": [], "c_dim": [], "vf_quality": []}

# model training
################

with torch.no_grad():
    for n in range(len(params["trial"])):

        alpha = params["alpha"][n]
        rep = params["trial"][n]
        print(f"Testing model fit {n+1} out of {n_trials} (alpha = {alpha}, rep = {rep})")

        # initialize rnn matrices
        bias = torch.tensor(params["bias"][n], device=device, dtype=dtype)
        W_in = torch.tensor(params["W_in"][n], device=device, dtype=dtype)
        L = torch.tensor(params["L"][n], device=device, dtype=dtype)
        R = torch.tensor(params["R"][n], device=device, dtype=dtype)
        W_r = torch.tensor(params["W_out"][n], device=device, dtype=dtype)

        # model initialization
        rnn = LowRankCRNN(torch.empty(size=(N, N), dtype=dtype, device=device), L, R, W_in, bias,
                          g="ReLU")

        # add noise to input
        inputs = [inp[:, :] + noise_lvl * np.random.randn(inp.shape[0], inp.shape[1]) for inp in inputs]

        # add dendritic gain controllers
        for key, c in zip(unique_conditions, params["conceptors"][n]):
            rnn.z_controllers[key] = torch.tensor(c, device=device, dtype=dtype)

        # set up loss function
        loss_func = torch.nn.MSELoss()

        # generate predictions
        test_loss, predictions, dynamics = [], [], []
        with torch.no_grad():
            for trial in range(train_trials, trials):

                # get input and target timeseries
                inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
                target = torch.tensor(targets[trial], device=device, dtype=dtype)

                # get random initial state
                rnn.activate_z_controller(conditions[trial])
                for step in range(init_steps):
                    x = torch.randn(n_in, dtype=dtype, device=device)
                    rnn.forward(x)
                rnn.detach()

                # make prediction
                y_col, z_col = [], []
                for step in range(steps):
                    if step > auto_steps:
                        inp[step, :n_out] = y
                    z = rnn.forward(inp[step])
                    y = W_r @ z
                    y_col.append(y)
                    z_col.append(z)
                predictions.append(np.asarray([y.detach().cpu().numpy() for y in y_col]))
                dynamics.append(np.asarray([z.detach().cpu().numpy() for z in z_col]))

                # calculate loss
                loss = loss_func(torch.stack(y_col, dim=0), target)
                test_loss.append(loss.item())

        # calculate conceptor dimensionality
        conceptors = [c.detach().cpu().numpy() for c in rnn.z_controllers.values()]
        c_dim = [np.sum(c**2) for c in conceptors]

        # save results
        results["alpha"].append(alpha)
        results["trial"].append(rep)
        results["train_loss"].append(params["train_loss"][n][-1] / batch_size)
        results["test_loss"].append(np.mean(test_loss))
        results["c_dim"].append(c_dim)
        #results["z_dim"].append(...)
        #results["vf_quality"].append(...)

        # save results
        pickle.dump(results, open(save_file, "wb"))

        if visualize_results:

            # prediction figure
            fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
            for i, trial in enumerate(np.random.choice(test_trials, size=(plot_examples,))):
                ax = axes[i]
                ax.plot(targets[train_trials + trial], label="targets", linestyle="dashed")
                for j, line in enumerate(ax.get_lines()):
                    ax.plot(predictions[trial][:, j], label="predictions", linestyle="solid", color=line.get_color())
                ax.axvline(x=auto_steps, color="grey", linestyle="dotted")
                ax.set_ylabel("amplitude")
                ax.set_title(f"test trial {trial + 1}")
                if i == plot_examples - 1:
                    ax.set_xlabel("steps")
            fig.suptitle("Model Predictions")
            plt.tight_layout()

            # dynamics figure
            fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2 * plot_examples))
            n_neurons = 5
            for i in range(plot_examples):
                mean_v = np.mean(dynamics[i], axis=-1)
                ax = axes[i]
                ax.plot(mean_v, color="black", label="mean")
                for j in np.random.choice(k, size=(n_neurons,)):
                    ax.plot(dynamics[i][:, j], label=f"neuron {j + 1}")
                ax.axvline(x=int(0.5 * steps), color="grey", linestyle="dashed")
                ax.set_ylabel("amplitude")
                if i == plot_examples - 1:
                    ax.set_xlabel("steps")
                    ax.legend()
            fig.suptitle("RNN dynamics")
            plt.tight_layout()

            # dimensionality figure
            conceptors = np.asarray([c.detach().cpu().numpy() for c in rnn.z_controllers.values()])
            fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
            ax = axes[0]
            ax.bar(np.arange(n_conditions)+0.25, c_dim, width=0.4, color="royalblue")
            ax.set_xlabel("task conditions")
            ax.set_ylabel("dim(c)")
            ax.set_title("Conceptor Dimensionalities")
            ax = axes[1]
            im = ax.imshow(conceptors, aspect="auto", interpolation="none", cmap="cividis")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("neurons")
            ax.set_ylabel("task conditions")
            ax.set_title("Conceptors")
            plt.tight_layout()

            # loss figure
            fig, ax = plt.subplots(figsize=(6, 4))
            condition_losses = []
            test_conditions = np.asarray(conditions[train_trials:])
            for c in unique_conditions:
                idx = np.argwhere(test_conditions == c).squeeze()
                condition_losses.append(np.mean(np.asarray(test_loss)[idx]))
            ax.bar(x=np.arange(len(unique_conditions)), height=condition_losses)
            ax.set_xlabel("conditions")
            ax.set_ylabel("MSE")
            ax.set_xticks(np.arange(len(unique_conditions)), labels=[f"{c}" for c in unique_conditions])
            ax.set_title("Test Loss")
            plt.tight_layout()

            plt.show()
