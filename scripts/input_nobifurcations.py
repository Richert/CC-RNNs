import numpy as np
import pickle
import matplotlib.pyplot as plt


def vanderpol(y: np.ndarray, x: float = 1.0, tau: float = 1.0) -> np.ndarray:
    y1, y2 = y[0], y[1]
    y1_dot = y2 / tau
    y2_dot = (y2*x*(1 - y1**2) - y1) / tau
    return np.asarray([y1_dot, y2_dot])


def pitchfork(y: np.ndarray, x: float = 1.0, tau: float = 5.0) -> np.ndarray:
    y1, y2 = y[0], y[1]
    y1_dot = (x*y1 - y1**3) / tau
    y2_dot = -y2 / tau
    return np.asarray([y1_dot, y2_dot])


# general parameters
save_path = f"/home/richard-gast/Documents/data"

# task parameters
trials = 10000
mu = 1.0
d = 1
dt = 0.01
sampling_rate = 10
steps = 10000
init_scale = 2.0
dim = 2

# plot parameters
plot_examples = 6
visualize = True

# define conditions
rhs_funcs = {1: pitchfork, 2: vanderpol}

# generate targets and inputs
targets, inputs, conditions = [], [], []
for n in range(trials):
    successful = False
    ds = np.random.choice(list(rhs_funcs))
    rhs = rhs_funcs[ds]
    while not successful:
        y = init_scale * np.random.randn(dim)
        y_col = []
        for step in range(steps + d):
            y = y + dt * rhs(y, x=mu)
            if step % sampling_rate == 0:
                y_col.append(y)
            if not np.isfinite(y_col[-1][0]):
                break
        y_col = np.asarray(y_col)
        if np.isfinite(y_col[-1, 0]):
            successful = True
    inputs.append(y_col[:-d])
    targets.append(y_col[d:])
    conditions.append(ds)
    print(f"Finished trial {n+1} of {trials}.")

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions},
            open(f"{save_path}/nobifurcations_2ds.pkl", "wb"))

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
