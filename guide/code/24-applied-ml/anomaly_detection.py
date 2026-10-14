# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/16-anomaly-detection
"""Anomaly detection from scratch: z-score, robust (MAD) z-score, IQR fences, k-NN distance, and an isolation forest.

All data is SYNTHETIC and seeded: two Gaussian clusters of normal points plus uniformly scattered anomalies.
Labels exist only so we can measure the detectors; the detectors never see them.
Run from guide/:  uv run code/24-applied-ml/anomaly_detection.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

EULER = 0.5772156649


# ---------- 1-D statistical rules ----------
def z_scores(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def robust_z(x):
    """Modified z-score: 0.6745 * (x - median) / MAD. 0.6745 makes it match z for Gaussian data."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / mad


def iqr_fences(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


# ---------- k-NN distance ----------
def knn_score(X_ref, X, k=10, same=True):
    """Distance to the k-th nearest reference point. If same, skip the zero distance to itself."""
    d = np.sqrt(((X[:, None, :] - X_ref[None, :, :]) ** 2).sum(-1))
    d.sort(axis=1)
    return d[:, k] if same else d[:, k - 1]


# ---------- isolation forest ----------
def c(n):
    """Average path length of an unsuccessful BST search over n points; normalizes depths."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2 * (math.log(n - 1) + EULER) - 2 * (n - 1) / n


def build_tree(X, depth, max_depth, rng):
    """Split on a random feature at a random value until a point is alone or depth runs out."""
    if depth >= max_depth or len(X) <= 1:
        return {"size": len(X)}
    f = rng.integers(X.shape[1])
    lo, hi = X[:, f].min(), X[:, f].max()
    if lo == hi:
        return {"size": len(X)}
    v = rng.uniform(lo, hi)
    left = X[:, f] < v
    return {"f": f, "v": v,
            "l": build_tree(X[left], depth + 1, max_depth, rng),
            "r": build_tree(X[~left], depth + 1, max_depth, rng)}


def path_length(x, node, depth=0):
    """Edges from root to the leaf containing x, plus c(size) for the unbuilt subtree below a leaf."""
    if "size" in node:
        return depth + c(node["size"])
    child = node["l"] if x[node["f"]] < node["v"] else node["r"]
    return path_length(x, child, depth + 1)


def isolation_forest(X, n_trees=100, sample=256, seed=0):
    """Returns s(x) = 2^(-E[h(x)] / c(sample)) for every row: near 1 = anomaly, about 0.5 or less = normal."""
    rng = np.random.default_rng(seed)
    psi = min(sample, len(X))
    max_depth = math.ceil(math.log2(psi))
    trees = [build_tree(X[rng.choice(len(X), psi, replace=False)], 0, max_depth, rng) for _ in range(n_trees)]
    mean_h = np.array([np.mean([path_length(x, t) for t in trees]) for x in X])
    return 2 ** (-mean_h / c(psi)), mean_h


def make_data(seed=0):
    """SYNTHETIC: 600 + 380 normal points in two clusters of different spread, 20 uniform anomalies (2%)."""
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], 0.6, (600, 2))
    b = rng.normal([4, 3], 1.2, (380, 2))
    out = rng.uniform([-4, -4], [9, 8], (60, 2))
    far = out[np.min(np.c_[np.linalg.norm(out - [0, 0], axis=1) / 0.6,
                           np.linalg.norm(out - [4, 3], axis=1) / 1.2], axis=1) > 3.5][:20]
    X = np.vstack([a, b, far])
    y = np.r_[np.zeros(980), np.ones(len(far))].astype(int)
    return X, y


def precision_at(scores, y, k):
    top = np.argsort(-scores)[:k]
    return float(y[top].mean())


if __name__ == "__main__":
    print("=== Worked example: one outlier masks itself ===")
    x = np.array([10, 11, 12, 12, 13, 14, 50], dtype=float)
    print(f"x = {x.astype(int).tolist()}")
    print(f"mean = {x.mean():.2f}, std = {x.std():.2f}, z(50) = {z_scores(x)[-1]:.2f}  -> NOT flagged at |z| > 3")
    print(f"median = {np.median(x):.0f}, MAD = {np.median(np.abs(x - np.median(x))):.0f}, "
          f"robust z(50) = {robust_z(x)[-1]:.2f}  -> flagged at |z| > 3.5")
    lo, hi = iqr_fences(x)
    print(f"Q1 = {np.percentile(x, 25)}, Q3 = {np.percentile(x, 75)}, fences = [{lo}, {hi}]  -> 50 flagged")
    assert abs(z_scores(x)[-1]) < 3 and robust_z(x)[-1] > 3.5

    print("\n=== Isolation forest normalizer c(n) ===")
    for n in [2, 16, 256]:
        print(f"c({n}) = {c(n):.3f}")
    for h in [4.0, c(256), 14.0]:
        print(f"E[h] = {h:6.3f}  ->  s = 2^(-{h:.3f}/{c(256):.3f}) = {2 ** (-h / c(256)):.3f}")

    X, y = make_data()
    print(f"\n=== SYNTHETIC 2-D data: {len(X)} points, {y.sum()} anomalies ({y.mean():.1%}) ===")
    scores = {}
    zmax = np.max(np.abs((X - X.mean(0)) / X.std(0)), axis=1)
    scores["max |z| per feature"] = zmax
    scores["k-NN distance (k=10)"] = knn_score(X, X, k=10)
    if_scratch, mean_h = isolation_forest(X)
    scores["isolation forest (scratch)"] = if_scratch
    skl = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
    scores["IsolationForest (sklearn)"] = -skl.score_samples(X)
    lof = LocalOutlierFactor(n_neighbors=20).fit(X)
    scores["LOF (k=20)"] = -lof.negative_outlier_factor_
    k = int(y.sum())
    for name, s in scores.items():
        print(f"  {name:28s} AUPRC = {average_precision_score(y, s):.3f}   precision@{k} = {precision_at(s, y, k):.2f}")
    print(f"  random scores AUPRC would be about the anomaly rate = {y.mean():.3f}")
    print(f"  scratch IF mean path length: anomalies {mean_h[y == 1].mean():.2f}, normals {mean_h[y == 0].mean():.2f}")
    assert average_precision_score(y, if_scratch) > average_precision_score(y, zmax)

    print("\n=== Threshold from an alert budget ===")
    budget = 30
    thr = np.sort(if_scratch)[-budget]
    flagged = if_scratch >= thr
    tp = int((flagged & (y == 1)).sum())
    print(f"flag top {budget}: threshold s >= {thr:.3f}, caught {tp} of {y.sum()} anomalies, "
          f"precision = {tp / flagged.sum():.2f}, recall = {tp / y.sum():.2f}")
    print(f"z-rule |z| > 3 on any feature flags {(zmax > 3).sum()} points, of which {int(((zmax > 3) & (y == 1)).sum())} are anomalies")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    gx, gy = np.meshgrid(np.linspace(-4.5, 9.5, 120), np.linspace(-4.5, 11.5, 120))
    grid_scores = -skl.score_samples(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    cs = ax.contourf(gx, gy, grid_scores, levels=12, cmap="Purples")
    fig.colorbar(cs, ax=ax, label="anomaly score (higher = stranger)")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=6, color=BLUE, alpha=0.6, label="normal")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=40, color=ORANGE, edgecolor="black", label="anomaly (synthetic label)")
    ax.set(title="Isolation forest score", xlabel="feature 1", ylabel="feature 2", ylim=(-4.5, 11.5))
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], s=6, color=GRAY, alpha=0.5)
    zf = zmax > 3
    top_if = if_scratch >= thr
    ax.scatter(X[zf, 0], X[zf, 1], s=70, facecolor="none", edgecolor=TEAL, lw=1.5, label=f"|z| > 3 ({zf.sum()} flagged)")
    ax.scatter(X[top_if, 0], X[top_if, 1], s=18, color=PURPLE, label=f"isolation forest top {budget}")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=80, marker="x", color=ORANGE, label="true anomaly")
    ax.set(title="What each rule flags", xlabel="feature 1", ylim=(-4.5, 11.5))
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[2]
    for (name, s), col in zip(scores.items(), [GRAY, TEAL, PURPLE, BLUE, ORANGE]):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=col, lw=1.5, label=f"{name}: {average_precision_score(y, s):.3f}")
    ax.axhline(y.mean(), color=GRAY, ls=":", lw=1)
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall (AUPRC in legend)", ylim=(0, 1.05))
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "anomaly-detection")
