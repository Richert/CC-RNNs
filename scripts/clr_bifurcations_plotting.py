import numpy as np
import pickle
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# load data
path = "/home/richard-gast/Documents/results"
task = "clr_bifurcations"

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

results = {"lambda": [], "trial": [], "repetition": [], "train_epochs": [], "train_loss": [], "test_loss": [], "mu": [],
           "c_dim": []}
conceptors = []
data = pickle.load(open(f"{path}/{task}_zfit.pkl", "rb"))

# sweep data
for point in range(len(data["trial"])):
    for trial in range(data["trial"][point]):
        results["lambda"].append(data["lambda"][point])
        results["trial"].append(trial)
        results["repetition"].append(data["trial"][point][trial])
        results["train_epochs"].append(data["train_epochs"][point][trial])
        results["train_loss"].append(data["train_loss"][point])
        results["test_loss"].append(data["test_loss"][point][trial])
        results["mu"].append(data["mu"][point][trial])
        results["c_dim"].append(data["c_dim"][point][trial])
        conceptors.append(data["conceptor"][point][trial])
df = DataFrame.from_dict(results)
conceptors = np.asarray(conceptors)

# plotting
##########

# create figure
fig = plt.figure()
grid = fig.add_gridspec(nrows=2, ncols=4)

# plot number of training epochs
ax = fig.add_subplot(grid[0, 0])
sb.barplot(df, x="lambda", y="train_epochs", hue="condition")
ax.set_title("Number of training epochs")
ax.set_ylabel("epochs")
ax.set_xlabel(r"$\lambda$")

# plot combined test loss
ax = fig.add_subplot(grid[0, 1])
sb.barplot(df, x="lambda", y="test_loss", hue="condition")
ax.set_title("Average prediction performance")
ax.set_ylabel("MSE")
ax.set_xlabel(r"$\lambda$")

# plot test loss as a function of mu
unique_conditions = np.unique(df.loc[:, "condition"].values)
for i, c in enumerate(unique_conditions):
    df_tmp = df.loc[df.loc[:, "condition"] == c, :]
    ax = fig.add_subplot(grid[0, 2+i])
    sb.lineplot(df_tmp, x="mu", y="test_loss", hue="lambda")
    ax.set_title(f"Prediction performance for {c}")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("MSE")

# plot conceptor dimensionalities
ax = fig.add_subplot(grid[1, 0])
sb.barplot(df, x="lambda", y="c_dim", hue="condition")
ax.set_title("Conceptor dimensionalities")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("dim(C)")

# plot example conceptors for each condition
unique_lambdas = np.unique(df.loc[:, "lambda"].values)[[0, 2, 4]]
for i, lam in enumerate(unique_lambdas):
    idx1 = df.loc[:, "lambda"] == lam
    cs = []
    for cond in unique_conditions:
        idx2 = df.loc[:, "condition"] == cond
        idx = np.argmin(df.loc[idx1 & idx2, "test_loss"].values)
        cs.append(conceptors[idx1 & idx2][idx])
    ax = fig.add_subplot(grid[1, 1+i])
    ax.imshow(np.asarray(cs), aspect="auto", interpolation="none", cmap="cividis")
    ax.set_ylabel("condition")
    ax.set_xlabel("soma")
    ax.set_title(rf"Conceptors for $\lambda = {lam}$")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'/home/richard-gast/Documents/results/clr_bifurcations.svg')
plt.show()
