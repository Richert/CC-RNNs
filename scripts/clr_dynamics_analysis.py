import numpy as np
import pickle
from pandas import DataFrame
import matplotlib.pyplot as plt

# load data
path = "/home/richard-gast/Documents/"
load_file = f"{path}/data/clr_dynamics.pkl"
save_file = f"{path}/results/clr_dynamics.csv"
data = pickle.load(open(load_file, "rb"))

# prepare data frame
lyapunov = np.zeros((len(data["trial"]),))
memory = np.zeros((len(data["trial"])))
columns = list(data.keys())
for key in ["z", "z_p"]:
    columns.pop(columns.index(key))
df = DataFrame(columns=columns + ["lyapunov", "memory"], index=np.arange(0, len(lyapunov)))

# calculate lyapunov exponent
for n in range(len(lyapunov)):

    z1, z2 = data["z"][n], data["z_p"][n]
    d0 = np.sqrt(np.sum((z1[0] - z2[0])**2))
    d1 = np.sqrt(np.sum((z1[-1] - z2[-1])**2))
    le = np.log(d1/d0) / len(z1)

    for c in columns:
        df.loc[n, c] = data[c][n]
    df.loc[n, "lyapunov"] = le

# calculate memory capacity
z1s, z2s = [], []
for n in range(len(lyapunov)):

    z, z_p = data["z"][n], data["z_p"][n]
    if data["input"][n]:
        z2s.append((z, z_p))
    else:
        z1s.append((z, z_p))

# save results
df.to_csv(save_file)
