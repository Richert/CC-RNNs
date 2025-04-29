import numpy as np
import pickle
import matplotlib.pyplot as plt


def vanderpol(y: np.ndarray, x: float = 1.0, tau: float = 1.0) -> np.ndarray:
    y1, y2 = y[0], y[1]
    y1_dot = y2 / tau
    y2_dot = (y2*x*(1 - y1**2) - y1) / tau
    return np.asarray([y1_dot, y2_dot])

def pitchfork(y: np.ndarray, x: float = 1.0, tau: float = 5.0) -> np.ndarray:
    y_dot = (x*y - y**3) / tau
    return y_dot

def lorenz(y: np.ndarray, s: float = 10.0, r: float = 28.0, b: float = 2.667) -> np.ndarray:
    y1, y2, y3 = y
    y1_dot = s*(y2 - y1)
    y2_dot = r*y1 - y2 - y1*y3
    y3_dot = y1*y2 - b*y3
    return np.asarray([y1_dot, y2_dot, y3_dot])


# general parameters
save_path = f"/home/richard-gast/Documents/data"

# task parameters
trials = 10000
min_mu, max_mu = -1.0, 1.0
ds_dims = [1]
n_conditions = len(ds_dims)
d = 1
dt = 0.01
sampling_rate = 5
steps = 5000
init_scale = 2.0

# plot parameters
plot_examples = 6
visualize = True

# define conditions
rhs_funcs = {1: pitchfork, 2: vanderpol, 3: lorenz}
inp_channels = {1: [0], 2: [1, 2], 3: [3, 4, 5]}

# generate targets and inputs
targets, inputs, conditions = [], [], []
for n in range(trials):
    successful = False
    dim = np.random.choice(ds_dims)
    rhs = rhs_funcs[dim]
    while not successful:
        y = init_scale * np.random.randn(dim)
        y_col = []
        for step in range(steps + d):
            y = y + dt * rhs(y)
            if step % sampling_rate == 0:
                y_col.append(y)
            if not np.isfinite(y_col[-1][0]):
                break
        y_col = np.asarray(y_col)
        if np.isfinite(y_col[-1, 0]):
            successful = True
    inp = np.zeros((y_col.shape[0], sum(ds_dims)))
    inp[:, inp_channels[dim]] = y_col
    inputs.append(inp[:-d, :])
    targets.append(inp[d:, :])
    conditions.append(dim)
    print(f"Finished trial {n+1} of {trials}.")

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions},
            open(f"{save_path}/mixed_{n_conditions}ds.pkl", "wb"))

# plot results
if visualize:
    fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
    for i, trial in enumerate(np.random.choice(trials, size=(plot_examples,))):
        ax = axes[i]
        # ax.plot(inputs[trial], label="x")
        ax.plot(targets[trial])
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.set_title(f"training trial {trial+1}: dim = {conditions[trial]}")
        # ax.legend()
    fig.suptitle("Inputs (x) and Target Waveforms (y)")
    plt.tight_layout()
    plt.show()
