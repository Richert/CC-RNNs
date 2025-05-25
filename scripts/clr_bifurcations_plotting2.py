import pickle
import matplotlib.pyplot as plt

# load data
path = "/home/richard-gast/Documents/results"
task = "simulations_bifurcations"

# matplotlib settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# collect data
data = pickle.load(open(f"{path}/{task}_zfit2.pkl", "rb"))

# best model fit: 93

# plotting
##########

for i in data.keys():

    # create figure
    fig = plt.figure()
    grid = fig.add_gridspec(nrows=len(data[i]["mu"]), ncols=1)

    # plot PF dynamics
    for j, mu in enumerate(data[i]["mu"]):
        ax = fig.add_subplot(grid[j, 0])
        ax.plot(data[i]["y"][j])
        ax.set_title(rf"$\mu = {mu}$")
        ax.set_xlabel(r"steps")
        ax.set_ylabel(r"$y$")

    # padding
    fig.suptitle(f"Model fit {i+1}")
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)

    # saving/plotting
    fig.canvas.draw()
    plt.savefig(f'{path}/{task}_{i+1}.svg')
    plt.show()
