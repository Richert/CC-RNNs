import numpy as np
import pickle
import matplotlib.pyplot as plt


def vanderpol(y: np.ndarray, x: float = 1.0, tau: float = 1.0) -> np.ndarray:
    y1, y2 = y[0], y[1]
    y1_dot = y2 / tau
    y2_dot = (y2*x*(1 - y1**2) - y1) / tau
    return np.asarray([y1_dot, y2_dot])


# general parameters
save_path = f"/home/richard-gast/Documents/data"

# task parameters
trials = 10000
mus = [0.5]
taus = [1.0, 2.0, 4.0]
n_conditions = len(taus)
d = 1
dt = 0.01
sampling_rate = 20
steps = 10000
init_scale = 2.0

# plot parameters
plot_examples = 6
visualize = True

# generate targets and inputs
y0 = 1.0
targets, inputs, conditions = [], [], []
for n in range(trials):
    successful = False
    tau = np.random.choice(taus)
    mu = np.random.choice(mus)
    while not successful:
        y = init_scale * np.random.randn(2)
        y_col = []
        for step in range(steps + d):
            y = y + dt * vanderpol(y, tau=tau)
            if step % sampling_rate == 0:
                y_col.append(y)
            if not np.isfinite(y_col[-1][0]):
                break
        y_col = np.asarray(y_col)
        if np.isfinite(y_col[-1, 0]):
            successful = True
    inputs.append(y_col[:-d])
    targets.append(y_col[d:])
    conditions.append((mu, tau))
    print(f"Finished trial {n+1} of {trials}.")

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions},
            open(f"{save_path}/vdp_{n_conditions}freqs.pkl", "wb"))

# plot results
if visualize:
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
