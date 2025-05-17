import sys
sys.path.append("../")
from src.rnn import LowRankCRNN
from src.functions import init_weights, init_dendrites
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# parameter definition
######################

# general
dtype = torch.float64
device = "cuda:0"
state_vars = ["y"]
path = "/home/richard"
load_file = f"{path}/data/bifurcations_2ds.pkl"
save_file = f"{path}/results/clr_bifurcations_zfit.pkl"
visualize_results = False
plot_examples = 6

# load inputs and targets
data = pickle.load(open(load_file, "rb"))
inputs = data["inputs"]
targets = data["targets"]
conditions = data["trial_conditions"]
unique_conditions = np.unique(conditions, axis=0)
n_conditions = len(unique_conditions)

# task parameters
steps = inputs[0].shape[0]
init_steps = 20
auto_steps = 100
noise_lvl = 0.0

# rnn parameters
k = 100
n_dendrites = 10
n_in = inputs[0].shape[-1]
n_out = targets[0].shape[-1]
density = 0.5
in_scale = 0.1
out_scale = 0.2
Delta = 0.1
sigma = 0.9
N = int(k * n_dendrites)

# training parameters
trials = len(conditions)
train_trials = int(0.9 * trials)
batch_size = 20
test_trials = int(0.5*batch_size)
augmentation = 1.0
lr = 1e-2
betas = (0.9, 0.999)
gradient_cutoff = 1e10
truncation_steps = 100
epsilon = 1.0
alpha = 15.0
batches = int(augmentation * train_trials / batch_size)

# sweep parameters
lambdas = [2e-4, 3e-4, 4e-4, 5e-4, 6e-4]
n_reps = 20
n_trials = len(lambdas)*n_reps

# prepare results
results = {"lambda": [], "trial": [], "train_epochs": [], "train_loss": [], "srl_loss": [], "conceptors": [],
           "L": [], "R": [], "W_in": [], "bias": [], "W_out": []}

# model training
################

n = 0
for rep in range(n_reps):
    for lam in lambdas:

        print(f"Starting run {n+1} out of {n_trials} training runs (lambda = {lam}, rep = {rep})")

        # initialize rnn matrices
        bias = torch.tensor(Delta * np.random.randn(k), device=device, dtype=dtype)
        W_in = torch.tensor(in_scale * np.random.randn(N, n_in), device=device, dtype=dtype)
        L = init_weights(N, k, density)
        W, R = init_dendrites(k, n_dendrites)
        W_r = torch.tensor(out_scale * np.random.randn(n_out, k), device=device, dtype=dtype, requires_grad=True)

        # model initialization
        rnn = LowRankCRNN(torch.tensor(W*0.0, dtype=dtype, device=device),
                          torch.tensor(L*sigma, dtype=dtype, device=device),
                          torch.tensor(R, device=device, dtype=dtype),
                          W_in, bias, g="ReLU", alpha=alpha, lam=lam)
        rnn.free_param("L")

        # add noise to input
        inputs = [inp[:, :] + noise_lvl * np.random.randn(inp.shape[0], inp.shape[1]) for inp in inputs]

        # initialize controllers
        for c in unique_conditions:
            rnn.init_new_z_controller(init_value="random")
            rnn.store_z_controller(c)

        # set up loss function
        loss_func = torch.nn.MSELoss()

        # set up optimizer
        optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
        rnn.clip(gradient_cutoff)

        # training
        train_loss = 0.0
        loss_col, slr_loss = [], []
        for batch in range(batches):

            loss = torch.zeros((1,), device=device, dtype=dtype)
            with torch.enable_grad():

                for trial in np.random.choice(train_trials, size=(batch_size,), replace=False):

                    # get input and target timeseries
                    inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
                    target = torch.tensor(targets[trial], device=device, dtype=dtype)

                    # get initial state
                    rnn.activate_z_controller(conditions[trial])
                    for step in range(init_steps):
                        x = torch.randn(n_in, dtype=dtype, device=device)
                        rnn.forward(x)
                    rnn.detach()

                    # collect loss
                    y_col = []
                    for step in range(steps):
                        z = rnn.forward(inp[step])
                        delta_c = rnn.update_z_controller()
                        y = W_r @ z
                        if step % truncation_steps == truncation_steps - 1:
                            rnn.detach()
                        y_col.append(y)
                        slr_loss.append((delta_c**2).sum().detach().cpu().numpy())

                    # calculate loss
                    y_col = torch.stack(y_col, dim=0)
                    loss += loss_func(y_col, target)

                    # store controller
                    rnn.store_z_controller(conditions[trial])

                # make update
                optim.zero_grad()
                loss.backward()
                optim.step()
                rnn.detach()

                # store and print loss
                train_loss = loss.item()
                loss_col.append(train_loss)

            # generate predictions
            test_loss, predictions, dynamics = [], [], []
            with torch.no_grad():
                for trial in range(test_trials):

                    # get input and target timeseries
                    inp = torch.tensor(inputs[train_trials + trial], device=device, dtype=dtype)
                    target = torch.tensor(targets[train_trials + trial], device=device, dtype=dtype)

                    # get initial state
                    rnn.activate_z_controller(conditions[train_trials + trial])
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

                test_loss_sum = np.sum(test_loss)
                print(f"Training epoch {batch + 1} / {batches}: Train MSE = {loss_col[-1]}, Test MSE = {test_loss_sum}")
                if test_loss_sum < epsilon:
                    break

        # calculate conceptor dimensionality
        conceptors = [c.detach().cpu().numpy() for c in rnn.z_controllers.values()]
        c_dim = [np.sum(c**2) for c in conceptors]

        # save results
        results["lambda"].append(lam)
        results["trial"].append(rep)
        results["train_epochs"].append(batch)
        results["train_loss"].append(loss_col)
        results["srl_loss"].append(slr_loss)
        results["conceptors"].append(conceptors)
        results["L"].append(rnn.L.detach().cpu().numpy())
        results["R"].append(R)
        results["bias"].append(bias.detach().cpu().numpy())
        results["W_in"].append(W_in.detach().cpu().numpy())
        results["W_out"].append(W_r.detach().cpu().numpy())

        # report progress
        n += 1
        print(f"Finished after {batch + 1} training epochs. Final loss: {loss_col[-1]}.")
        print(f"Conceptor dimensions: {c_dim}")

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
            fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
            ax = axes[0]
            ax.plot(loss_col)
            ax.set_xlabel("training batch")
            ax.set_ylabel("MSE")
            ax.set_title("Training loss")
            ax = axes[1]
            ax.plot(gaussian_filter1d(slr_loss, sigma=500))
            ax.set_xlabel("training batch")
            ax.set_ylabel("delta")
            ax.set_title("Conceptor loss")
            # ax = axes[2]
            # condition_losses = []
            # test_conditions = np.asarray(conditions[train_trials:])
            # for c in unique_conditions:
            #     idx = np.argwhere(test_conditions == c).squeeze()
            #     condition_losses.append(np.mean(np.asarray(test_loss)[idx]))
            # ax.bar(x=np.arange(len(unique_conditions)), height=condition_losses)
            # ax.set_xlabel("conditions")
            # ax.set_ylabel("MSE")
            # ax.set_xticks(np.arange(len(unique_conditions)), labels=[f"{c}" for c in unique_conditions])
            # ax.set_title("Test Loss")
            plt.tight_layout()

            plt.show()
