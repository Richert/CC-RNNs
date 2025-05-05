import numpy as np
import pickle
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
path = "/home/richard-gast/Documents/results"
task = "clr_rhythmic_3freqs"
conditions = ["fit", "cfit", "cfit_noweights"]
taus = [1.0, 2.0, 3.0]

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12.0, 7.0)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# collect data
##############

results = {"condition": [], "sigma": [], "Delta": [], "trial": [], "train_epochs": [], "train_loss": [],
           "test_loss": [], "loss_tau1": [], "loss_tau2": [], "loss_tau3": []}
examples = {}

for cond in conditions:

    data = pickle.load(open(f"{path}/{task}_{cond}.pkl", "rb"))

    # tau-specific losses
    test_conds, test_loss = data.pop("test_conditions"), data["test_loss"]
    for conds, losses in zip(test_conds, test_loss):
        for tau in taus:
            l = np.asarray([l for l, c in zip(losses, conds) if c[1] == 1.0])
            results[f"loss_tau{int(tau)}"].append(np.mean(l))

    # sweep data
    for key, val in data.items():
        if type(val[0]) is list:
            val = [np.mean(v) for v in val]
        if key in results:
            results[key].extend(val)
    results["condition"].extend([cond]*len(data["trial"]))

    # example fits
    examples[cond] = pickle.load(open(f"{path}/{task}_{cond}_single.pkl", "rb"))

df = DataFrame.from_dict(results)

# plotting
##########

# create figure
fig = plt.figure()
grid = fig.add_gridspec(nrows=5, ncols=4)

# plot number of training epochs
ax = fig.add_subplot(grid[:2, 0])
sb.lineplot(df, x="sigma", y="train_epochs", hue="condition")
ax.set_title("Number of training epochs")
ax.set_ylabel("epochs")
ax.set_xlabel(r"$\sigma$")

# plot examples of training progress
ax = fig.add_subplot(grid[:2, 1])
for cond in conditions:
    ax.plot(examples[cond]["train_loss"], label=cond)
ax.set_xlabel("training epoch")
ax.set_ylabel("MSE")
ax.set_title(r"Training progress for $\sigma = 0.6$")
ax.legend()

# plot combined test loss
ax = fig.add_subplot(grid[:2, 2])
sb.lineplot(df, x="sigma", y="test_loss", hue="condition")
ax.set_title("Average prediction performance")
ax.set_ylabel("MSE")
ax.set_xlabel(r"$\sigma$")

# plot example test loss for specific conditions
ax = fig.add_subplot(grid[:2, 3])
cond_results = {"architecture": [], "tau": [], "MSE": []}
sigma = 0.6
df_tmp = df.loc[df.loc[:, "sigma"] == sigma, :]
for cond in conditions:
    df_tmp2 = df_tmp.loc[df_tmp.loc[:, "condition"] == cond, :]
    for tau in taus:
        loss = df_tmp2.loc[:, f"loss_tau{int(tau)}"].values.tolist()
        cond_results["architecture"].extend([cond] * len(loss))
        cond_results["tau"].extend([tau] * len(loss))
        cond_results["MSE"].extend(loss)
df_tmp = DataFrame.from_dict(cond_results)
sb.barplot(df_tmp, x="tau", y="MSE", hue="architecture")
ax.set_title(rf"Prediction performance for $\sigma = {sigma}$")
ax.set_xlabel(r"$\tau$")

# plot example time series for each condition
unique_conditions = np.unique(examples[conditions[0]]["conditions"], axis=0).tolist()
cond_colors = ["royalblue", "darkorange", "darkgreen"]
for j, c1 in enumerate(unique_conditions):
    ax = fig.add_subplot(grid[2+j, :])
    for i, c2 in enumerate(conditions):
        data = examples[c2]
        test_conds, targs, preds = data["conditions"], data["targets"], data["predictions"]
        idx = test_conds.index(tuple(c1))
        if i == 0:
            ax.plot(targs[idx][:, 0], color="black", label="target")
        ax.plot(preds[idx][:, 0], color=cond_colors[i], label=c2)
    ax.axvline(x=100, linestyle="dashed", color="grey")
    ax.set_ylabel("y")
    ax.set_title(rf"Predictions for $\tau = {c1[1]}$")
    if j == 2:
        ax.set_xlabel("steps")
        ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'/home/richard-gast/Documents/results/clr_rhythmic_fitting.svg')
plt.show()
