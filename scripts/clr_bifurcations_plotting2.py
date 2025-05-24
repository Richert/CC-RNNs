import pickle
import matplotlib.pyplot as plt

# load data
path = "/home/richard/results"
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
data = pickle.load(open(f"{path}/{task}_zfit.pkl", "rb"))

# plotting
##########

for i in range(len(data["pf"])):

    # create figure
    fig = plt.figure()
    grid = fig.add_gridspec(nrows=3, ncols=1)

    # plot PF dynamics
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(data["pf"][i])
    ax.set_title("Pitchfork bifurcation")
    ax.set_xlabel(r"steps")
    ax.set_ylabel(r"$z$")

    # plot PF dynamics
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(data["vdp"][i])
    ax.set_title("Hopf bifurcation")
    ax.set_xlabel(r"steps")
    ax.set_ylabel(r"$z$")

    # DS switching
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(data["pf_vdp"][i])
    ax.set_title("Dynamical systems switching")
    ax.set_xlabel(r"steps")
    ax.set_ylabel(r"$z$")

    # padding
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)

    # saving/plotting
    fig.canvas.draw()
    plt.savefig(f'{path}/{task}_{i+1}.svg')
    # plt.show()
