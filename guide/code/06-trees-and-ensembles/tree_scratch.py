# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/04-decision-trees
"""A decision tree from scratch in NumPy: Gini impurity, exhaustive split search, recursive growth,
prediction. Then the same tree in scikit-learn. Data is SYNTHETIC (noisy two moons, NumPy only).

Run from guide/:  uv run code/06-trees-and-ensembles/tree_scratch.py
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def gini(y):
    """1 - sum_k p_k^2 for the integer labels in y. 0 = pure node."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p**2)


def best_split(X, y, min_leaf=1):
    """Try every feature and every midpoint between sorted unique values; keep the biggest Gini decrease."""
    best = (None, None, 0.0)                      # (feature, threshold, decrease)
    parent = gini(y)
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        for t in (values[:-1] + values[1:]) / 2:  # midpoints
            left = X[:, j] <= t
            n_l, n_r = left.sum(), (~left).sum()
            if n_l < min_leaf or n_r < min_leaf:
                continue
            child = (n_l * gini(y[left]) + n_r * gini(y[~left])) / len(y)
            if parent - child > best[2]:
                best = (j, t, parent - child)
    return best


def grow(X, y, depth=0, max_depth=None, min_leaf=1):
    """Split until the node is pure, max_depth is hit, or no split lowers impurity. Leaves store the majority class."""
    leaf = {"value": int(np.bincount(y).argmax())}
    if gini(y) == 0 or (max_depth is not None and depth >= max_depth):
        return leaf
    j, t, gain = best_split(X, y, min_leaf)
    if j is None:
        return leaf
    mask = X[:, j] <= t
    return {"feature": j, "threshold": t,
            "left": grow(X[mask], y[mask], depth + 1, max_depth, min_leaf),
            "right": grow(X[~mask], y[~mask], depth + 1, max_depth, min_leaf)}


def predict_one(node, x):
    while "feature" in node:                      # walk down until a leaf
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["value"]


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


def count_leaves(node):
    return 1 if "feature" not in node else count_leaves(node["left"]) + count_leaves(node["right"])


def make_moons(n, noise, rng):
    """Two interleaving half circles, label 0 and 1, plus Gaussian noise. SYNTHETIC."""
    a = rng.uniform(0, np.pi, n // 2)
    b = rng.uniform(0, np.pi, n - n // 2)
    X = np.r_[np.c_[np.cos(a), np.sin(a)], np.c_[1 - np.cos(b), 0.5 - np.sin(b)]]
    y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    return X + rng.normal(0, noise, X.shape), y


if __name__ == "__main__":
    # The lesson's worked example: 10 customers, two candidate splits on one node.
    y_node = np.array([0] * 5 + [1] * 5)
    for name, left in [("age < 35", [0, 0, 0, 0, 1]), ("income < 50k", [0, 0, 0, 1, 1])]:
        left = np.array(left)
        right = 1 - left                              # mirror image: [1,4] or [2,3]
        child = (5 * gini(left) + 5 * gini(right)) / 10
        print(f"{name}: weighted Gini {child:.2f}, decrease {gini(y_node) - child:.2f}")

    rng = np.random.default_rng(0)
    X, y = make_moons(600, 0.3, rng)
    idx = rng.permutation(len(y))
    tr, te = idx[:300], idx[300:]

    print("\nFrom scratch on two moons (300 train / 300 test):")
    for d in (1, 3, 6, None):
        tree = grow(X[tr], y[tr], max_depth=d)
        acc_tr = np.mean(predict(tree, X[tr]) == y[tr])
        acc_te = np.mean(predict(tree, X[te]) == y[te])
        print(f"  max_depth={str(d):4s} leaves={count_leaves(tree):3d} train {acc_tr:.3f} test {acc_te:.3f}")

    print("\nscikit-learn, same data:")
    for d in (1, 3, 6, None):
        sk = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X[tr], y[tr])
        print(f"  max_depth={str(d):4s} leaves={sk.get_n_leaves():3d} "
              f"train {sk.score(X[tr], y[tr]):.3f} test {sk.score(X[te], y[te]):.3f}")

    # Sanity checks: the hand numbers, and a pure node never splits.
    assert abs(gini(np.array([0, 0, 0, 0, 1])) - 0.32) < 1e-12
    assert "feature" not in grow(X[:5], np.zeros(5, int))
