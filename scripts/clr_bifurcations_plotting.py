import numpy as np
import pickle
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
path = "/home/richard/results"
task = "test_bifurcations"

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (10.0, 7.0)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# collect data
##############

results = {"condition": [], "lambda": [], "trial": [], "repetition": [], "train_epochs": [],
           "srl_loss": [], "test_loss": [], "mu": [], "c_dim": []}
conceptors, predictions, targets = [], [], []
data = pickle.load(open(f"{path}/{task}_zfit.pkl", "rb"))

# sweep data
for trial in range(len(data["test_loss"])):
    results["trial"].append(trial % 20)
    results["condition"].append("PF" if data["condition"][trial] == 1 else "VDP")
    results["lambda"].append(data["lambda"][trial])
    results["repetition"].append(data["repetition"][trial])
    results["train_epochs"].append(data["train_epochs"][trial])
    results["srl_loss"].append(float(data["srl_loss"][trial]))
    results["test_loss"].append(data["test_loss"][trial])
    results["mu"].append(data["mu"][trial])
    results["c_dim"].append(data["c_dim"][trial])
    conceptors.append(data["conceptor"][trial])
    predictions.append(data["predictions"][trial])
    targets.append(data["targets"][trial])
df = DataFrame.from_dict(results)
conceptors = np.asarray(conceptors)
predictions = np.asarray(predictions)
targets = np.asarray(targets)

# plotting
##########

# create figure
fig = plt.figure()
grid = fig.add_gridspec(nrows=4, ncols=12)

# plot conceptor loss
ax = fig.add_subplot(grid[0, :3])
sb.barplot(df, x="lambda", y="srl_loss", hue="condition")
ax.set_title("Conceptor loss")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("loss(C)")

# plot test loss
ax = fig.add_subplot(grid[0, 3:6])
sb.barplot(df, x="lambda", y="test_loss", hue="condition")
ax.set_title("Average prediction performance")
ax.set_ylabel("MSE")
ax.set_xlabel(r"$\lambda$")

# plot test loss as a function of mu
unique_conditions = np.unique(df.loc[:, "condition"].values)
for i, c in enumerate(unique_conditions):
    df_tmp = df.loc[df.loc[:, "condition"] == c, :]
    ax = fig.add_subplot(grid[0, 3*(i+2):3*(i+3)])
    sb.lineplot(df_tmp, x="mu", y="test_loss", hue="lambda")
    ax.set_title(f"Prediction performance for {c} system")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("MSE")

# plot conceptor dimensionalities
ax = fig.add_subplot(grid[1, :3])
sb.barplot(df, x="lambda", y="c_dim", hue="condition")
ax.set_title("Conceptor dimensionalities")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("dim(C)")

# plot example conceptors and time series for each condition
lam = 4e-4
mus = [-0.1, 0.0, 0.1]
idx1 = df.loc[:, "lambda"] == lam
cs = []
for i, cond in enumerate(unique_conditions):
    idx2 = df.loc[:, "condition"] == cond
    for j, mu in enumerate(mus):
        df_tmp = df.loc[idx1 & idx2, :]
        unique_mus = np.unique(df_tmp.loc[:, "mu"].values)
        m = unique_mus[np.argmin(np.abs(unique_mus - mu)).squeeze()]
        idx3 = df_tmp.loc[:, "mu"] == m
        losses = df_tmp.loc[idx3, "test_loss"].values
        min_idx = np.argmin(losses)
        ax = fig.add_subplot(grid[2+i, 4*j:4*(j+1)])
        targs = targets[idx1 & idx2][idx3][min_idx]
        preds = predictions[idx1 & idx2][idx3][min_idx]
        for k in range(targs.shape[1]):
            l = ax.plot(targs[:, k], label=f"target {k+1}", linestyle="dashed")
            ax.plot(preds[:, k], label=f"prediction {k+1}", linestyle="solid", color=l[0].get_color())
        ax.set_xlabel("steps")
        ax.set_ylabel("y")
        ax.set_title(rf"Predictions for {cond} system with $\mu = {np.round(m, decimals=2)}$")
        ax.legend()
    cs.append(conceptors[idx1 & idx2][idx3][min_idx])
ax = fig.add_subplot(grid[1, 3:])
im = ax.imshow(np.asarray(cs), aspect="auto", interpolation="none", cmap="cividis")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_yticks([0, 1], labels=unique_conditions)
ax.set_ylabel("condition")
ax.set_xlabel("soma")
ax.set_title(rf"Conceptors for $\lambda = {lam}$")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'{path}/{task}.svg')
# plt.show()
