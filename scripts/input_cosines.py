import numpy as np
import pickle
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# task parameters
trials = 10000
omega = 6.0
sigma = 10.0
theta = np.cos(0.2*np.pi/omega)
steps = 100
dt = 0.01
time = np.arange(steps) * dt

# condition parameters
freqs = [(1.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
phases = [(0.0, 0.0), (0.0, np.pi), (0.0 ,np.pi)]
n_conditions = len(freqs)

# plot parameters
plot_examples = 6
visualize = True

# generate targets and inputs
targets, inputs, conditions = [], [], []
for n in range(trials):
    c = np.random.choice(n_conditions)
    y1 = np.sin(2*np.pi*time*omega*freqs[c][0] + phases[c][0])
    y2 = np.sin(2*np.pi*time*omega*freqs[c][1] + phases[c][1])
    inputs.append(sigmoid(sigma*(y1 - theta)))
    targets.append(np.asarray([y1, y2]).T)
    conditions.append(c)

# save results
pickle.dump({"inputs": inputs, "targets": targets, "trial_conditions": conditions},
            open(f"../data/cosine_inputs_omega{int(omega)}.pkl", "wb"))

# plot results
fig, axes = plt.subplots(nrows=plot_examples, figsize=(12, 2*plot_examples))
for i, trial in enumerate(np.random.choice(trials, size=(plot_examples,))):
    ax = axes[i]
    ax.plot(time, inputs[trial], label="x")
    ax.plot(time, targets[trial][:, 0], label="y1")
    ax.plot(time, targets[trial][:, 1], label="y2")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.set_title(f"training trial {trial+1}: condition = {conditions[trial]}")
    ax.legend()
fig.suptitle("Inputs (x) and Target Waveforms (y)")
plt.tight_layout()
plt.show()
