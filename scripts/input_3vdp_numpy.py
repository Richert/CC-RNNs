import numpy as np
import pickle
import matplotlib.pyplot as plt


def vanderpol(y1: np.ndarray, y2: np.ndarray, x: float, tau: float = 1.0) -> tuple:
    y1_dot = y2 / tau
    y2_dot = (y2*x*(1 - y1**2) - y1) / tau
    return y1_dot, y2_dot

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# task parameters
trials = 10000
mu = 0.5
min_tau = 0.2
sigma = 2.0
theta = 0.0
d = 1
dt = 0.01
sampling_rate = 20
steps = 2000

# condition parameters
y0 = 2.0
init_conditions = {0: (-y0, y0), 1: (y0, -y0), 2: (0.0, y0)}

# plot parameters
plot_examples = 6
visualize = True

# generate targets and inputs
targets, inputs, conditions = [], [], []
for n in range(trials):
    successful = False
    tau = (1.0 - min_tau) * np.random.rand() + min_tau
    c = np.random.choice(len(init_conditions))
    while not successful:
        y1 = np.zeros((2,)) - y0
        y2 = np.zeros((2,)) + y0
        y1[0], y2[0] = init_conditions[c][0], init_conditions[c][1]
        x_col, y_col = [], []
        for step in range(steps + d):
            y1_dot, y2_dot = vanderpol(y1, y2, x=mu, tau=tau)
            y1 = y1 + dt * y1_dot
            y2 = y2 + dt * y2_dot
            if step % sampling_rate == 0:
                y_col.append(y1)
                x_col.append(sigmoid(sigma*(y1[1] - theta)))
        y_col = np.asarray(y_col)
        x_col = np.asarray(x_col)
        if np.isfinite(y_col[-1, 0]):
            successful = True
    inputs.append(x_col[:-d])
    targets.append(y_col[d:])
    conditions.append(c)

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions},
            open(f"../data/3vdp_inputs_sr{sampling_rate}.pkl", "wb"))

# plot results
fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
for i, trial in enumerate(np.random.choice(trials, size=(plot_examples,))):
    ax = axes[i]
    ax.plot(inputs[trial], label="x")
    ax.plot(targets[trial][:, 0], label="y1")
    ax.plot(targets[trial][:, 1], label="y2")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.set_title(f"training trial {trial+1}: in-phase = {conditions[trial]}")
    ax.legend()
fig.suptitle("Inputs (x) and Target Waveforms (y)")
plt.tight_layout()
plt.show()
