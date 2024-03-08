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


def minmax(x: np.ndarray) -> np.ndarray:
    x = x - np.min(x)
    x = x / np.max(x)
    return x


# parameter definitions
#######################

# plotting parameters
plot_lorenz = True
plot_recall = True

# lorenz equation parameters
s_col = [10.0, 12.0, 20.0]
r_col = [10.0, 14.0, 28.0]
b_col = [0.5, 1.2, 2.667]
dt = 0.01
steps = 100000
lorenz_indices = [0, 1, 2]

# reservoir parameters
N = 500
alpha = 200.0
sr = 1.5
bias = 0.9
in_scale = 1.2
density = 1.0
init_steps = 1000
tychinovs = (0.001, 0.001)

# training parameters
recall_steps = 2000

# obtain Lorenz dynamics
########################

n_in = len(lorenz_indices)

# simulation
lorenz_states = []
for s, r, b in zip(s_col, r_col, b_col):

    y = np.asarray([0.1, 0.9, 1.1])
    y_col = []
    for step in range(steps + init_steps):
        y = y + dt * lorenz(y[0], y[1], y[2], s=s, r=r, b=b)
        y_col.append([y[idx] for idx in lorenz_indices])

    y_col = np.asarray(y_col)
    for i, idx in enumerate(lorenz_indices):
        y_col[:, i] = minmax(y_col[:, i])
    lorenz_states.append(y_col)

# plotting
lorenz_vars = ["x", "y", "z"]
if plot_lorenz:

    fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
    for i, (ax, y_col) in enumerate(zip(axes, lorenz_states)):
        for j, idx in enumerate(lorenz_indices):
            ax.plot(y_col[init_steps:init_steps+recall_steps, j], label=lorenz_vars[idx])
        if i == 2:
            ax.set_xlabel("steps")
            ax.legend()
    plt.suptitle("Lorenz dynamics (input)")
    plt.tight_layout()

# train conceptors
##################

# initialize reservoir
r = Reservoir(N, n_in=n_in, alpha=alpha, sr=sr, bias_scale=bias, inp_scale=in_scale, density=density)

# drive reservoir with each input pattern and learn a conceptor for replicating it
r_states, _ = r.run(lorenz_states, learning_steps=steps, init_steps=init_steps, load_reservoir=True,
                    tychinov_alphas=tychinovs)

# recall training patterns
recall = r.recall(r_states[:, :, -1], recall_steps=recall_steps)

# plot recall patterns
recall_indices = [0, 1]
if plot_recall:

    fig2, axes = plt.subplots(nrows=3, figsize=(12, 6))
    for i, (ax, y_recall) in enumerate(zip(axes, recall)):
        for j, idx in enumerate(lorenz_indices):
            ax.plot(y_recall[:, j], label=lorenz_vars[idx])
        if i == 2:
            ax.set_xlabel("steps")
            ax.legend()
        plt.suptitle("RNN recall")
    plt.tight_layout()
    plt.show()


