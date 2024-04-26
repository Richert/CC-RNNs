import pickle
import matplotlib.pyplot as plt

# load data
conditions = ["lr", "rfc", "lr_rfc"]
data = {}
for cond in conditions:
    res = pickle.load(open(f"../results/{cond}_lorenz.pkl", "rb"))
    if "lorenz" not in data:
        data["lorenz"] = res["targets"]
    data[cond] = res["predictions"]

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
plot_start = 1000
plot_stop = 9000

# plotting
##########

# create figure layout
fig = plt.figure(layout="constrained", figsize=(12, 10))
subfigs = fig.subfigures(nrows=4)

# plot lorenz dynamics
combinations = [[0, 1], [0, 2], [1, 2]]
titles = [r"\textbf{(A)} Lorenz equations", r"\textbf{(B)} LR-RNN", r"\textbf{(C)} RFC", r"\textbf{(D)} CC-LR-RNN"]
colors = ["black", "darkgreen", "darkblue", "darkred"]
for i, c in enumerate(["lorenz"] + conditions):

    fig_tmp = subfigs[i]
    axes = fig_tmp.subplots(ncols=3)

    for j in range(len(conditions)):

        ax = axes[j]
        s = data[c]
        idx = combinations[j]

        ax.plot(s[plot_start:plot_stop, idx[0]], s[plot_start:plot_stop, idx[1]], color=colors[i])
        ax.set_xlabel(fr"$u_{idx[0]+1}$")
        ax.set_ylabel(fr"$u_{idx[1]+1}$")

    fig_tmp.suptitle(titles[i])

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.02)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'../results/lorenz.svg')
plt.show()
