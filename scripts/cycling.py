import numpy as np
import matplotlib.pyplot as plt
from src import Reservoir

# parameter definitions
#######################

# input parameters
fmin = 2.0
fmax = 10.0
steps = 200000
init_steps = 1000
dt = 1e-3

# reservoir parameters
N = 400
alpha = 400.0
sr = 1.5
bias = 0.9
in_scale = 1.2
density = 0.5
tychinovs = (0.001, 0.0001)

# training parameters
recall_steps = 2000

# train conceptors
##################

# generate periodic input patterns
time = np.arange(steps+init_steps)*dt
p1 = np.zeros((steps+init_steps, 1))
p1[:, 0] = np.sin(time*fmin*np.pi*2.0)
p2 = np.zeros_like(p1)
p2[:, 0] = np.sin(time*fmax*np.pi*2.0)

# initialize reservoir
r = Reservoir(N, n_in=1, alpha=alpha, sr=sr, bias_scale=bias, inp_scale=in_scale, density=density)

# train reservoir on patterns
states, _ = r.run([p1, p2], learning_steps=steps, init_steps=init_steps, load_reservoir=True,
                  tychinov_alphas=tychinovs)

# recall training patterns
recall = r.recall(states[:, :, -1], recall_steps=recall_steps)

# plotting
##########

fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

ax = axes[0]
ax.plot(p1[init_steps:init_steps+recall_steps, 0], label="target")
ax.plot(recall[0, :, 0], label="recall")
ax.set_title("Pattern I")
ax.legend()

ax = axes[1]
ax.plot(p2[init_steps:init_steps+recall_steps, 0], label="target")
ax.plot(recall[1, :, 0], label="recall")
ax.set_title("Pattern II")
ax.set_xlabel("steps")

plt.tight_layout()
plt.show()
