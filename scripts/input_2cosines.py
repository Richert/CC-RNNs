import numpy as np
import pickle
import matplotlib.pyplot as plt

# task parameters
trials = 10000
omegas = [4.0, 6.0, 8.0]
thetas = [0.0, np.pi]
inp_width = 1
steps = 100
dt = 0.01
time = np.arange(steps) * dt

# condition parameters
n_conditions = len(omegas)

# plot parameters
plot_examples = 6
visualize = True

# generate targets and inputs
targets, inputs, conditions = [], [], []
for n in range(trials):
    theta_1 = 2*np.pi*np.random.rand()
    omega_1 = np.random.choice(omegas)
    y_1 = np.sin(2*np.pi*time*omega_1 + theta_1)
    # theta_2 = np.random.choice(thetas)
    omega_2 = np.random.choice(omegas)
    y_2 = np.sin(2*np.pi*time*omega_2 + theta_1)
    inp = np.zeros_like(y_1)
    idx = np.argwhere(np.diff(1.0*(y_1 < 0) - 1.0*(y_1 > 0)) < -1.0)
    inp[idx] = 1.0
    inputs.append(inp)
    targets.append(np.asarray([y_1, y_2]).T)
    conditions.append((omega_1, omega_2))

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions, "time": time},
            open(f"../data/2cosines_inputs_{n_conditions}f.pkl", "wb"))

# plot results
fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
for i, trial in enumerate(np.random.choice(trials, size=(plot_examples,))):
    ax = axes[i]
    ax.plot(time, inputs[trial], label="x")
    ax.plot(time, targets[trial], label="y")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.set_title(f"training trial {trial+1}: condition = {conditions[trial]}")
    ax.legend()
fig.suptitle("Inputs (x) and Target Waveforms (y)")
plt.tight_layout()
plt.show()
