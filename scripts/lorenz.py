import numpy as np
import matplotlib.pyplot as plt
from src import Reservoir

# function definitions
######################


def lorenz(x: float, y: float, z: float, s: float = 10.0, r: float = 28.0, b: float = 2.667) -> np.ndarray:
    """
    Parameters
    ----------
    x, y, z: float
        State variables.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : np.ndarray
       Vectorfield of the Lorenz equations.
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.asarray([x_dot, y_dot, z_dot])


# parameter definitions
#######################

# plotting parameters
plot_lorenz = True
plot_recall = True

# lorenz equation parameters
s_col = [10.0, 12.0, 20.0]
r_col = [15.0, 21.0, 28.0]
b_col = [0.5, 1.2, 2.667]
dt = 0.01
steps = 10000

# reservoir parameters
N = 200
alpha = 10.0
sr = 1.2
bias = 0.2
in_scale = 1.5
density = 0.2
init_steps = 500
tychinovs = (0.01, 0.001)

# training parameters
recall_steps = 1000

# obtain Lorenz dynamics
########################

# simulation
lorenz_states = []
for s, r, b in zip(s_col, r_col, b_col):

    y = np.asarray([0.1, 0.9, 1.1])
    y_col = []
    for step in range(steps + init_steps):
        y = y + dt * lorenz(y[0], y[1], y[2], s=s, r=r, b=b)
        y_col.append(y[:])

    lorenz_states.append(np.asarray(y_col))

# plotting
if plot_lorenz:

    fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
    for i, (ax, y_col) in enumerate(zip(axes, lorenz_states)):
        ax.plot(y_col[:, 0], label="x")
        ax.plot(y_col[:, 1], label="y")
        ax.plot(y_col[:, 2], label="z")
        if i == 2:
            ax.set_xlabel("steps")
            ax.legend()
    plt.suptitle("Lorenz dynamics (input)")
    plt.tight_layout()
    plt.show()

# train conceptors
##################

# initialize reservoir
r = Reservoir(N, n_in=2, alpha=alpha, sr=sr, bias_scale=bias, inp_scale=in_scale, density=density)

# drive reservoir with each input pattern and learn a conceptor for replicating it
_ = r.run(lorenz_states, learning_steps=steps, init_steps=init_steps, load_reservoir=True, tychinov_alphas=tychinovs)

# recall training patterns
recall = r.recall(recall_steps=recall_steps)

# plot recall patterns
if plot_recall:

    fig2, axes = plt.subplots(nrows=3, figsize=(12, 6))
    for i, (ax, y_recall) in enumerate(zip(axes, recall)):
        ax.plot(y_recall[:, 0], label="x")
        ax.plot(y_recall[:, 1], label="y")
        ax.plot(y_recall[:, 2], label="z")
        if i == 2:
            ax.set_xlabel("steps")
            ax.legend()
        plt.suptitle("RNN recall")
    plt.tight_layout()
    plt.show()


