import numpy as np
import pickle
import matplotlib.pyplot as plt


def vanderpol(y1: np.ndarray, y2: np.ndarray, x: float, tau: float = 1.0) -> tuple:
    y1_dot = y2 / tau
    y2_dot = (y2*x*(1 - y1**2) - y1) / tau
    return y1_dot, y2_dot


# task parameters
trials = 10000
min_mu, max_mu = -1.0, 1.0
taus = [0.5, 1.0, 2.0]
n_conditions = len(taus)
d = 1
dt = 0.01
sampling_rate = 20
steps = 10000

# plot parameters
plot_examples = 6
visualize = True

# generate targets and inputs
y0 = 1.0
targets, inputs, conditions = [], [], []
for n in range(trials):
    successful = False
    tau = np.random.choice(taus)
    mu = (max_mu - min_mu) * np.random.rand() + min_mu
    while not successful:
        y1 = np.zeros((1,)) - y0
        y2 = np.zeros((1,)) + y0
        y_col = []
        for step in range(steps + d):
            y1_dot, y2_dot = vanderpol(y1, y2, x=mu, tau=tau)
            y1 = y1 + dt * y1_dot
            y2 = y2 + dt * y2_dot
            if step % sampling_rate == 0:
                y_col.append(y1)
        y_col = np.asarray(y_col)
        if np.isfinite(y_col[-1, 0]):
            successful = True
    inp = np.zeros((y_col.shape[0] - d, 2))
    inp[:, 0] = y_col[:-d]
    inp[:, 1] = mu
    inputs.append(inp)
    targets.append(y_col[d:])
    conditions.append(tau)

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions},
            open(f"../data/vdp_{n_conditions}freqs.pkl", "wb"))

# plot results
fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
for i, trial in enumerate(np.random.choice(trials, size=(plot_examples,))):
    ax = axes[i]
    ax.plot(inputs[trial], label="x")
    ax.plot(targets[trial], label="y")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.set_title(f"training trial {trial+1}: tau = {conditions[trial]}")
    ax.legend()
fig.suptitle("Inputs (x) and Target Waveforms (y)")
plt.tight_layout()
plt.show()
