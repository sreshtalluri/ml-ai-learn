# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/11-ensemble-methods
"""Ensembles from scratch in NumPy: a random forest (bootstrap rows + random feature per split + vote) and
gradient boosting with stumps (fit residuals, take a small step). Reuses the tree from tree_scratch.py.
Then the scikit-learn versions. Data is SYNTHETIC.

Run from guide/:  uv run code/06-trees-and-ensembles/ensembles_scratch.py
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

from tree_scratch import best_split, gini, grow, make_moons, predict


def grow_random(X, y, rng, max_features):
    """A forest tree: like grow(), but each split may only look at max_features random columns."""
    leaf = {"value": int(np.bincount(y, minlength=2).argmax())}
    if gini(y) == 0:
        return leaf
    cols = rng.choice(X.shape[1], max_features, replace=False)
    j, t, _ = best_split(X[:, cols], y)
    if j is None:
        return leaf
    j = cols[j]                                   # map back to the real column
    mask = X[:, j] <= t
    return {"feature": j, "threshold": t,
            "left": grow_random(X[mask], y[mask], rng, max_features),
            "right": grow_random(X[~mask], y[~mask], rng, max_features)}


def random_forest(X, y, n_trees, max_features, rng):
    """Bagging: each tree sees a bootstrap sample (n rows drawn with replacement)."""
    trees = []
    for _ in range(n_trees):
        rows = rng.integers(0, len(y), len(y))
        trees.append(grow_random(X[rows], y[rows], rng, max_features))
    return trees


def forest_predict(trees, X):
    votes = np.array([predict(t, X) for t in trees])   # [n_trees, n_samples]
    return (votes.mean(axis=0) > 0.5).astype(int)      # majority vote for labels 0/1


def fit_stump(x, r):
    """Depth-1 regression tree on one feature: the threshold whose two leaf means minimize squared error."""
    best = (np.inf, None, None, None)
    values = np.unique(x)
    for t in (values[:-1] + values[1:]) / 2:
        left, right = r[x <= t], r[x > t]
        sse = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
        if sse < best[0]:
            best = (sse, t, left.mean(), right.mean())
    _, t, lv, rv = best
    return lambda z: np.where(z <= t, lv, rv), t


def gradient_boost(x, y, rounds, lr):
    """F_0 = mean(y); each round fits a stump to the residuals y - F and adds lr * stump."""
    F = np.full(len(y), y.mean())
    history = [F.copy()]
    for _ in range(rounds):
        stump, _ = fit_stump(x, y - F)            # residual y - F = negative gradient of (y - F)^2 / 2
        F = F + lr * stump(x)
        history.append(F.copy())
    return history


if __name__ == "__main__":
    # Worked example from the lesson: x = [1,2,3,4], y = [2,4,7,9], stumps, learning rate 0.5.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 7.0, 9.0])
    for m, F in enumerate(gradient_boost(x, y, rounds=3, lr=0.5)):
        print(f"round {m}: F = {F.round(4).tolist()}  MSE {np.mean((y - F) ** 2):.4f}")

    sk = GradientBoostingRegressor(n_estimators=3, learning_rate=0.5, max_depth=1).fit(x[:, None], y)
    print(f"scikit-learn after 3 rounds: {sk.predict(x[:, None]).round(4).tolist()}")

    rng = np.random.default_rng(0)
    X, y_cls = make_moons(600, 0.3, rng)
    idx = rng.permutation(len(y_cls))
    tr, te = idx[:300], idx[300:]

    single = grow(X[tr], y_cls[tr])
    print(f"\nOne unlimited tree:       test {np.mean(predict(single, X[te]) == y_cls[te]):.3f}")
    for B in (1, 10, 100):
        trees = random_forest(X[tr], y_cls[tr], B, max_features=1, rng=np.random.default_rng(1))
        print(f"Scratch forest, {B:3d} trees: test {np.mean(forest_predict(trees, X[te]) == y_cls[te]):.3f}")
    rf = RandomForestClassifier(n_estimators=100, max_features=1, random_state=0).fit(X[tr], y_cls[tr])
    print(f"scikit-learn forest, 100:  test {rf.score(X[te], y_cls[te]):.3f}")

    # Sanity check: the hand-computed round-3 predictions.
    assert np.allclose(gradient_boost(x, y, 3, 0.5)[-1], [2.8125, 4.3125, 6.8125, 8.0625])
