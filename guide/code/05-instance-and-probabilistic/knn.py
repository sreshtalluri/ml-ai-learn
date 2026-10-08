"""K-nearest neighbors: worked example (raw vs standardized distances), from scratch, scikit-learn, figure.

Run from guide/:  uv run code/05-instance-and-probabilistic/knn.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Worked example: five customers, one query (age in years, income in $k)
# ---------------------------------------------------------------------------
names = ["A", "B", "C", "D", "E"]
X = np.array([[22, 54], [26, 60], [44, 90], [47, 30], [60, 57]], dtype=float)
y = np.array([0, 0, 1, 1, 1])
q = np.array([45, 56], dtype=float)


def knn_predict(X, y, q, k, metric="euclidean"):
    diff = X - q
    d = np.sqrt((diff**2).sum(axis=1)) if metric == "euclidean" else np.abs(diff).sum(axis=1)
    order = np.argsort(d, kind="stable")[:k]
    vote = Counter(y[order]).most_common(1)[0][0]
    return d, order, vote


print("Raw units")
d, order, vote = knn_predict(X, y, q, 3)
for n, xi, di in zip(names, X, d):
    print(f"  {n} {xi}  dx={xi[0]-q[0]:+.0f} dy={xi[1]-q[1]:+.0f}  euclid={di:.2f}")
print(f"  3 nearest: {[names[i] for i in order]} labels {y[order]} -> predict {vote}")

dm, order_m, _ = knn_predict(X, y, q, 3, "manhattan")
print(f"  manhattan: {dict(zip(names, dm.round(1)))} nearest {[names[i] for i in order_m]}")

mu, sigma = X.mean(axis=0), X.std(axis=0)  # population std, fitted on training rows only
Z, qz = (X - mu) / sigma, (q - mu) / sigma
print(f"\nStandardized (mean={mu.round(2)}, std={sigma.round(2)})  query z={qz.round(3)}")
dz, order_z, vote_z = knn_predict(Z, y, qz, 3)
for n, zi, di in zip(names, Z, dz):
    print(f"  {n} z={zi.round(3)}  dist={di:.3f}")
print(f"  3 nearest: {[names[i] for i in order_z]} labels {y[order_z]} -> predict {vote_z}")

# ---------------------------------------------------------------------------
# 2. SYNTHETIC dataset: purchase depends on age; income is mostly noise
# ---------------------------------------------------------------------------
rng = np.random.default_rng(5)
n = 300
age = rng.uniform(20, 70, n)
income = rng.uniform(30, 180, n)
p = 1 / (1 + np.exp(-(age - 45) / 4 + rng.normal(0, 0.6, n)))
label = (rng.uniform(size=n) < p).astype(int)
data = np.c_[age, income * 1000]  # income in dollars: same information, 1000x larger numbers

for k in (1, 15):
    raw = cross_val_score(KNeighborsClassifier(k), data, label, cv=5).mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), data, label, cv=5).mean()
    print(f"\nK={k:2d}  5-fold accuracy raw={raw:.3f}  standardized={scaled:.3f}")

# ---------------------------------------------------------------------------
# 3. Figure: decision regions
# ---------------------------------------------------------------------------
gx, gy = np.meshgrid(np.linspace(18, 72, 220), np.linspace(25, 185, 220))
grid = np.c_[gx.ravel(), gy.ravel() * 1000]
panels = [
    ("K = 1, standardized: jagged, overfits", make_pipeline(StandardScaler(), KNeighborsClassifier(1))),
    ("K = 15, standardized: smooth", make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
    ("K = 15, raw dollars: income dominates", KNeighborsClassifier(15)),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (title, model) in zip(axes, panels):
    model.fit(data, label)
    zz = model.predict(grid).reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.15)
    ax.scatter(age[label == 0], income[label == 0], s=12, color=BLUE, label="did not buy")
    ax.scatter(age[label == 1], income[label == 1], s=12, color=ORANGE, marker="s", label="bought")
    ax.set(title=title, xlabel="age (years)")
    ax.title.set_fontsize(10)
axes[0].set_ylabel("income ($k, synthetic)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "knn")
