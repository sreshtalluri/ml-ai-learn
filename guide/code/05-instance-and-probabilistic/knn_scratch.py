# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/06-knn-and-distances and phases/01-math-foundations/14-norms-and-distances
"""K-nearest neighbors from scratch (NumPy), checked against scikit-learn.

Run from guide/:  uv run code/05-instance-and-probabilistic/knn_scratch.py
"""
from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --- Chunk 1: distances from one query to every stored row --------------------
def distances(X, q, metric="euclidean"):
    diff = X - q                                    # [n, p] by broadcasting
    if metric == "euclidean":
        return np.sqrt((diff**2).sum(axis=1))       # [n]
    if metric == "manhattan":
        return np.abs(diff).sum(axis=1)
    if metric == "cosine":                          # 1 - cos(angle); ignores length
        norms = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
        return 1 - (X @ q) / np.maximum(norms, 1e-12)
    raise ValueError(metric)


# --- Chunk 2: standardization fitted on training rows only ---------------------
def fit_scaler(X_train):
    mu = X_train.mean(axis=0)
    sigma = np.maximum(X_train.std(axis=0), 1e-12)  # guard: constant feature
    return mu, sigma


def apply_scaler(X, mu, sigma):
    return (X - mu) / sigma                         # same mu, sigma for every query


# --- Chunk 3: find the K nearest and vote ---------------------------------------
def knn_predict(X_train, y_train, q, k=3, metric="euclidean", weighted=False):
    d = distances(X_train, q, metric)
    nearest = np.argsort(d, kind="stable")[:k]      # indices of the k smallest
    votes = Counter()
    for i in nearest:
        votes[y_train[i]] += 1 / max(d[i], 1e-12) if weighted else 1
    return votes.most_common(1)[0][0], nearest


# --- Worked example from the lesson: raw units vs standardized -----------------
names = np.array(["A", "B", "C", "D", "E"])
X = np.array([[22, 54], [26, 60], [44, 90], [47, 30], [60, 57]], dtype=float)
y = np.array([0, 0, 1, 1, 1])
q = np.array([45, 56], dtype=float)

pred, nn = knn_predict(X, y, q, k=3)
print(f"raw units:     nearest {names[nn].tolist()} -> predict {pred}")
mu, sigma = fit_scaler(X)
pred_z, nn_z = knn_predict(apply_scaler(X, mu, sigma), y, apply_scaler(q, mu, sigma), k=3)
print(f"standardized:  nearest {names[nn_z].tolist()} -> predict {pred_z}")


# --- Chunk 4: the curse of dimensionality --------------------------------------
rng = np.random.default_rng(0)
print("\ndims  nearest/farthest distance from a random query (uniform cube, 1000 points)")
for p in (2, 10, 100, 1000):
    pts = rng.uniform(size=(1000, p))
    d = distances(pts, rng.uniform(size=p))
    print(f"{p:5d}  {d.min() / d.max():.3f}")


# --- Use the library: same predictions as the scratch version ------------------
# SYNTHETIC: purchase depends on age, income is noise (same recipe as knn.py)
n = 300
age, income = rng.uniform(20, 70, n), rng.uniform(30, 180, n)
label = (rng.uniform(size=n) < 1 / (1 + np.exp(-(age - 45) / 4))).astype(int)
data = np.c_[age, income]
train, test = data[:240], data[240:]

mu, sigma = fit_scaler(train)
ours = np.array([knn_predict(apply_scaler(train, mu, sigma), label[:240], apply_scaler(t, mu, sigma), k=15)[0]
                 for t in test])
lib = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)).fit(train, label[:240]).predict(test)
print(f"\nscratch vs scikit-learn on 60 test rows: agree on {np.mean(ours == lib):.0%}, "
      f"test accuracy {np.mean(ours == label[240:]):.3f}")
assert pred == 0 and pred_z == 1
