"""Decision-tree splits by hand, tree depth vs overfitting, random forest vs single tree, and gradient boosting by hand.

Run from guide/:  uv run code/06-trees-and-ensembles/trees.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def gini(counts):
    counts = np.asarray(counts, float)
    p = counts / counts.sum()
    return 1 - np.sum(p**2)


def entropy(counts):
    p = np.asarray(counts, float) / np.sum(counts)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# 1. Choosing a split by Gini impurity
# ---------------------------------------------------------------------------
parent = [5, 5]
for name, left, right in [("age < 35", [4, 1], [1, 4]), ("income < 50k", [3, 2], [2, 3])]:
    nl, nr = sum(left), sum(right)
    weighted = nl / 10 * gini(left) + nr / 10 * gini(right)
    print(f"{name}: gini L={gini(left):.3f} R={gini(right):.3f} weighted={weighted:.3f} decrease={gini(parent) - weighted:.3f}"
          f" | entropy parent={entropy(parent):.3f} weighted={nl / 10 * entropy(left) + nr / 10 * entropy(right):.3f}")

# ---------------------------------------------------------------------------
# 2. Gradient boosting by hand: x = [1, 2, 3, 4], y = [2, 4, 7, 9], learning rate 0.5, stumps
# ---------------------------------------------------------------------------
x = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.0, 7.0, 9.0])
lr = 0.5
F = np.full(4, y.mean())
print(f"\nBoosting: F0 = mean(y) = {y.mean()}  MSE = {np.mean((y - F) ** 2):.4f}")
for m in range(1, 4):
    r = y - F
    stump = DecisionTreeRegressor(max_depth=1).fit(x, r)
    h = stump.predict(x)
    F = F + lr * h
    print(f"  round {m}: residuals {r.round(4).tolist()}  split x <= {stump.tree_.threshold[0]:.1f}  "
          f"h = {h.round(4).tolist()}  F = {F.round(4).tolist()}  MSE = {np.mean((y - F) ** 2):.4f}")

# ---------------------------------------------------------------------------
# 3. SYNTHETIC moons: depth, forests, boosting
# ---------------------------------------------------------------------------
X, yy = make_moons(n_samples=600, noise=0.3, random_state=2)
Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.5, random_state=0)
for depth in (1, 3, 6, None):
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    print(f"tree depth {str(depth):4s}: train {t.score(Xtr, ytr):.3f}  test {t.score(Xte, yte):.3f}")
rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=0).fit(Xtr, ytr)
print(f"random forest (300 trees): train {rf.score(Xtr, ytr):.3f}  test {rf.score(Xte, yte):.3f}")
print(f"gradient boosting (200 x depth 3, lr 0.05): train {gb.score(Xtr, ytr):.3f}  test {gb.score(Xte, yte):.3f}")

gx, gy = np.meshgrid(np.linspace(-2, 3, 250), np.linspace(-1.6, 2.1, 250))
grid = np.c_[gx.ravel(), gy.ravel()]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
models = [("Tree, depth 2", DecisionTreeClassifier(max_depth=2, random_state=0)),
          ("Tree, no depth limit", DecisionTreeClassifier(random_state=0)),
          ("Random forest, 300 trees", rf),
          ("Gradient boosting", gb)]
for ax, (title, mdl) in zip(axes, models):
    mdl.fit(Xtr, ytr)
    ax.contourf(gx, gy, mdl.predict_proba(grid)[:, 1].reshape(gx.shape), levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    ax.scatter(*Xtr[ytr == 0].T, s=7, color=BLUE)
    ax.scatter(*Xtr[ytr == 1].T, s=7, color=ORANGE)
    ax.set(title=f"{title}\ntrain {mdl.score(Xtr, ytr):.2f} · test {mdl.score(Xte, yte):.2f}", xlabel="x₁")
    ax.title.set_fontsize(9)
axes[0].set_ylabel("x₂")
save(fig, "decision-trees")

# Boosting figure: test accuracy vs number of trees for two learning rates
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for lr_, col in [(0.5, ORANGE), (0.05, TEAL)]:
    g = GradientBoostingClassifier(n_estimators=400, learning_rate=lr_, max_depth=3, random_state=0).fit(Xtr, ytr)
    axes[0].plot([np.mean(p == yte) for p in g.staged_predict(Xte)], color=col, label=f"test, lr {lr_}")
    axes[0].plot([np.mean(p == ytr) for p in g.staged_predict(Xtr)], color=col, ls="--", lw=1, label=f"train, lr {lr_}")
axes[0].set(title="Boosting: learning rate × number of trees", xlabel="number of trees", ylabel="accuracy", ylim=(0.8, 1.0))
axes[0].legend(frameon=False, fontsize=8)
ax = axes[1]
xs = np.linspace(0.5, 4.5, 200)[:, None]
F = np.full(4, y.mean())
Fs = np.full(200, y.mean())
ax.scatter(x.ravel(), y, color=BLUE, s=50, zorder=3, label="data")
ax.plot(xs, Fs, color=GRAY, label="F0 = mean")
for m, col in zip(range(1, 4), [PURPLE, ORANGE, TEAL]):
    stump = DecisionTreeRegressor(max_depth=1).fit(x, y - F)
    F = F + lr * stump.predict(x)
    Fs = Fs + lr * stump.predict(xs)
    ax.plot(xs, Fs, color=col, label=f"after round {m}")
ax.set(title="Gradient boosting on 4 points (lr 0.5)", xlabel="x", ylabel="y")
ax.legend(frameon=False, fontsize=8)
save(fig, "random-forests-and-boosting")
