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
load_file = f"{path}/data/bifurcations_2ds.pkl"
save_file = f"{path}/results/clr_bifurcations_cfit.pkl"
visualize_results = True
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
test_trials = trials - train_trials
augmentation = 1.0
lr = 1e-2
betas = (0.9, 0.999)
batch_size = 20
gradient_cutoff = 1e10
truncation_steps = 100
epsilon = 0.1
lam = 1e-5
batches = int(augmentation * train_trials / batch_size)

# sweep parameters
alphas = [3.0]
noise_lvls = [0.0]
n_reps = 10
n_trials = len(alphas)*n_reps

# prepare results
results = {"alpha": [], "noise": [], "trial": [], "train_epochs": [], "train_loss": [], "test_loss": [],
           "conceptors": [], "synapses": []}

# model training
################

n = 0
for rep in range(n_reps):
    for alpha in alphas:
        for noise_lvl in noise_lvls:

            print(f"Starting run {n+1} out of {n_trials} training runs (alpha = {alpha}, noise = {noise_lvl}, rep = {rep})")

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
                rnn.init_new_y_controller(init_value="random")
                rnn.store_y_controller(tuple(c))

            # set up loss function
            loss_func = torch.nn.MSELoss()

            # set up optimizer
            optim = torch.optim.Adam(list(rnn.parameters()) + [W_r], lr=lr, betas=betas)
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
                        rnn.activate_y_controller(conditions[trial])
                        for step in range(init_steps):
                            x = torch.randn(n_in, dtype=dtype, device=device)
                            rnn.forward(x)
                            rnn.update_y_controller()
                        rnn.detach()

                        # collect loss
                        y_col = []
                        for step in range(steps):
                            z = rnn.forward(inp[step])
                            rnn.update_y_controller()
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
                    rnn.detach()

                    # store and print loss
                    train_loss = loss.item()
                    loss_col.append(train_loss)
                    print(f"Training epoch {batch+1} / {batches}: MSE = {loss_col[-1]}")
                    if train_loss < epsilon:
                        break

            # generate predictions
            test_loss, predictions, dynamics = [], [], []
            with torch.no_grad():
                for trial in range(train_trials, trials):

                    # get input and target timeseries
                    inp = torch.tensor(inputs[trial], device=device, dtype=dtype)
                    target = torch.tensor(targets[trial], device=device, dtype=dtype)

                    # get initial state
                    rnn.activate_y_controller(conditions[trial])
                    for step in range(init_steps):
                        x = torch.randn(n_in, dtype=dtype, device=device)
                        rnn.forward(x)
                    rnn.detach()

                    # make prediction
                    y_col, z_col = [], []
                    for step in range(steps):
                        x = inp[step] if step <= auto_steps else y
                        z = rnn.forward(x)
                        y = W_r @ z
                        y_col.append(y)
                        z_col.append(z)
                    predictions.append(np.asarray([y.detach().cpu().numpy() for y in y_col]))
                    dynamics.append(np.asarray([z.detach().cpu().numpy() for z in z_col]))

                    # calculate loss
                    loss = loss_func(torch.stack(y_col, dim=0), target)
                    test_loss.append(loss.item())

            # save results
            results["alpha"].append(alpha)
            results["noise"].append(noise_lvl)
            results["trial"].append(rep)
            results["train_epochs"].append(batch)
            results["train_loss"].append(loss_col)
            results["test_loss"].append(np.sum(test_loss))
            results["conceptors"].append([c.detach().cpu().numpy() for c in rnn.y_controllers.values()])
            results["synapses"].append(rnn.L.detach().cpu().numpy())

            # report progress
            n += 1
            c_dim = [torch.mean(c).detach().cpu().numpy() for c in rnn.y_controllers.values()]
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

                # conceptors figure
                conceptors = np.asarray([c.detach().cpu().numpy() for c in rnn.y_controllers.values()])
                fig, ax = plt.subplots(figsize=(12, 2 * len(conceptors)))
                im = ax.imshow(conceptors, aspect="auto", interpolation="none", cmap="cividis")
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_xlabel("neurons")
                ax.set_ylabel("conditions")
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
                condition_losses = []
                test_conditions = np.asarray(conditions[train_trials:])
                for c in unique_conditions:
                    idx = np.argwhere(test_conditions == tuple(c)).squeeze()
                    condition_losses.append(np.mean(np.asarray(test_loss)[idx]))
                ax.bar(x=np.arange(len(unique_conditions)), height=condition_losses)
                ax.set_xlabel("conditions")
                ax.set_ylabel("MSE")
                ax.set_xticks(np.arange(len(unique_conditions)), labels=[f"{c}" for c in unique_conditions])
                ax.set_title("Test Loss")
                plt.tight_layout()

                plt.show()
