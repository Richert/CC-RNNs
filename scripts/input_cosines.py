import numpy as np
import pickle
import matplotlib.pyplot as plt

# task parameters
trials = 10000
omegas = [2.0]
inp_width = 1
steps = 200
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
    theta = 2*np.pi*np.random.rand()
    omega = np.random.choice(omegas)
    y = np.sin(2*np.pi*time*omega + theta)
    inp = np.zeros_like(y)
    idx = np.argwhere(np.diff(1.0*(y < 0) - 1.0*(y > 0)) < -1.0)
    inp[idx] = 1.0
    inputs.append(inp)
    targets.append(y)
    conditions.append(omega)

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions, "time": time},
            open(f"../data/cosine_inputs_{n_conditions}f.pkl", "wb"))

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
