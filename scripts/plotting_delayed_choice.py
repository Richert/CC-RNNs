import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os
import pandas as pd


# data collection
#################

files = [f for f in os.listdir("../results/lr") if f[:7] == "delayed"]
df = pd.DataFrame(columns=["noise", "delay", "rep", "train_error", "test_performance"],
                  index=np.arange(0, len(files)))
eps = 1e-10
n = 0

for file in files:

    # extract data
    data = pickle.load(open(f"../results/lr/{file}", "rb"))
    noise = data["condition"]["noise"]
    rep = data["condition"]["repetition"]
    error = data["training_error"]
    performance = data["classification_performance"]
    delay = int(file.split("_")[1][1:])

    df.loc[n, :] = (noise, delay, rep, error, performance)
    n += 1

# bring data into 2D matrix format
train_error = df.pivot_table(index="noise", columns="delay", values="train_error", aggfunc="mean")
test_perf = df.pivot_table(index="noise", columns="delay", values="test_performance", aggfunc="mean")
train_error = train_error[train_error.columns].astype(float)
test_perf = test_perf[test_perf.columns].astype(float)

# plotting
##########

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 8
plot_start = 2000
plot_stop = 7000

# figure layout
fig = plt.figure(figsize=(12, 5))
grid = fig.add_gridspec(ncols=2)

# training error
ax = fig.add_subplot(grid[0])
sb.heatmap(train_error, ax=ax)
ax.set_xlabel("delay period duration")
ax.set_ylabel("noise level")
ax.set_title("MSE (training data)")

# test performance
ax = fig.add_subplot(grid[1])
sb.heatmap(test_perf, ax=ax)
ax.set_xlabel("delay period duration")
ax.set_ylabel("noise level")
ax.set_title("Classification performance (test data)")

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.02)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'../results/delayed_choice.svg')
plt.show()
