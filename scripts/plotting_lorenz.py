import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os
import pandas as pd


# data collection and analysis
##############################

files = [f for f in os.listdir("../results/lr") if f[0] == "lorenz"]
models = ["lr", "clr", "rfc_k10", "rfc_k20", "rfc_k40", "rfc_k80", "rfc_k160", "rfc_k320", "rfc_k640"]
df = pd.DataFrame(columns=["model", "noise", "rep", "wd", "k_star", "dim", "train_error"],
                  index=np.arange(0, len(files)))
eps = 1e-10
n = 0
for file in files:
    for model in models:

        # extract data
        data = pickle.load(open(f"../results/{model}/{file}", "rb"))
        noise = data["condition"]["noise"]
        rep = data["condition"]["repetition"]
        error = np.mean(data["training_error"].detach().cpu().numpy())
        wd = data["wd"]

        # get dimensionality
        try:
            k_star = data["k_star"]
            dim = np.sum(data["c"] > eps)
        except KeyError:
            k_star = data["config"]["k"]
            dim = k_star

        df.loc[n, :] = (model, noise, rep, wd, k_star, dim, error)
        n += 1

# collect representative trajectories for target steps
noise = [0.0, 0.04, 0.1]
trajectories = {model: [] for model in models}
for lvl in noise:

    df_tmp = df.loc[np.round(df.loc[:, "noise"], decimals=2) == lvl, :]

    for model in models:

        # find representative sample
        df_tmp2 = df_tmp.loc[df_tmp.loc[:, "model"] == model, :]
        error = np.mean(df_tmp2.loc[:, "train_error"].values)
        idx = np.argmin(np.abs(df_tmp2.loc[:, "train_error"].values - error))
        rep = df_tmp2.loc[df_tmp2.index[idx], "rep"]

        # load data
        data = pickle.load(open(f"../results/{model}/lorenz_noise{int(lvl*100)}_{int(rep)}.pkl", "rb"))
        trajectories[model].append(data["predictions"])

# get target trajectory
targets = pickle.load(open(f"../results/lr/{files[0]}", "rb"))["targets"]

# plotting
##########

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 8
plot_start = 2000
plot_stop = 7000

# figure layout
fig = plt.figure(figsize=(12, 8))
subfigs = fig.subfigures(nrows=2, height_ratios=[0.3, 0.7])

# line plots
x = "noise"
ys = ["wd", "train_error"]
titles = ["Wasserstein Distance (test data)", "MSE (training data)"]
axes = subfigs[0].subplots(ncols=len(ys))
for i in range(len(ys)):

    ax = axes[i]
    sb.lineplot(x=x, y=ys[i], hue="model", data=df, color="0.8", err_style="bars", errorbar="sd", ax=ax)
    ax.set_xlabel("noise level")
    ax.set_ylabel(titles[i])

subfigs[0].suptitle("Model performance")

# trajectory plots
titles = [f"noise = {n}" for n in noise] + ["Lorenz equations"]
colors = ["darkgreen", "darkred"]
model_titles = ["LR", "LRI"]
lvars = [0, 2]
axes = subfigs[1].subplots(ncols=len(titles), nrows=len(models))
for i in range(len(models)):
    for j in range(len(titles)):

        ax = axes[i, j]
        if j < len(titles) - 1:
            data = trajectories[models[i]][j]
            c = colors[i]
        else:
            data = targets
            c = "black"

        ax.plot(data[plot_start:plot_stop, lvars[0]], data[plot_start:plot_stop, lvars[1]], color=c)
        ax.set_xlabel(fr"$u_{lvars[0] + 1}$")
        ax.set_ylabel(fr"$u_{lvars[1] + 1}$")
        ax.set_title(f"{model_titles[i]}: {titles[j]}")

subfigs[1].suptitle("Lorenz attractor reconstructions")

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.02)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'../results/lorenz.svg')
plt.show()
