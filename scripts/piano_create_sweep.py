import pickle
import sys
path = sys.argv[-3]
sys.path.append(path)

# define condition
keys = int(sys.argv[-2])
fingers = int(sys.argv[-1])

# load data
data = pickle.load(open(f"~/PycharmProjects/CC-RNNs/data/piano/piano_crnn_{keys}keys_{fingers}fingers.pkl", "rb"))

# define sweep
alphas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
motifs = [2, 4, 6, 8, 10]
motif_lengths = [2, 4, 6, 8, 10]
sweep = []
for alpha in alphas:
    for m in motifs:
        for l in motif_lengths:
            sweep.append((alpha, m, l))

# save data
data["sweep"] = sweep
pickle.dump(data, open(f"~/PycharmProjects/CC-RNNs/data/piano/piano_crnn_{keys}keys_{fingers}fingers.pkl", "wb"))
