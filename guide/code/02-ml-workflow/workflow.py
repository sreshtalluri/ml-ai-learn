"""Vocabulary, splits, leakage, and the bias-variance trade-off on synthetic data.

Run from guide/:  uv run code/02-ml-workflow/workflow.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. Vocabulary figure: features, target, prediction, residual
# ---------------------------------------------------------------------------
sqft = rng.uniform(0.8, 3.0, 25)                         # SYNTHETIC houses
price = 50 + 120 * sqft + rng.normal(0, 25, 25)
m = LinearRegression().fit(sqft[:, None], price)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(sqft, price, color=BLUE, s=24, zorder=3, label="samples: (feature x, target y)")
g = np.linspace(0.7, 3.1, 50)
ax.plot(g, m.predict(g[:, None]), color=TEAL, lw=2, label=f"model ŷ = {m.coef_[0]:.0f}x + {m.intercept_:.0f}  (parameters)")
i = int(np.argmax(np.abs(price - m.predict(sqft[:, None]))))
ax.annotate("", xy=(sqft[i], m.predict([[sqft[i]]])[0]), xytext=(sqft[i], price[i]), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
ax.text(sqft[i] + 0.05, (price[i] + m.predict([[sqft[i]]])[0]) / 2, "residual y − ŷ\n(squared → loss)", color=ORANGE, fontsize=9)
ax.scatter([2.6], m.predict([[2.6]]), marker="*", s=200, color=PURPLE, zorder=4, label="inference: prediction for a new house")
ax.set(title="The objects of a supervised problem", xlabel="feature: size (thousand sq ft)", ylabel="target: price ($k)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "ml-vocabulary")
print(f"vocabulary: w={m.coef_[0]:.1f}, b={m.intercept_:.1f}")

# ---------------------------------------------------------------------------
# 2. Leakage demo: scaling fit on all data vs inside the pipeline is small;
#    a target-derived feature is huge.
# ---------------------------------------------------------------------------
n = 2000
X = rng.normal(size=(n, 5))
y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1.5, n) > 0).astype(int)
leak = y + rng.normal(0, 0.3, n)                         # e.g. "refund_issued": recorded AFTER the outcome
Xl = np.c_[X, leak]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xltr, Xlte, _, _ = train_test_split(Xl, y, test_size=0.25, random_state=0, stratify=y)
clean = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr).score(Xte, yte)
leaky = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xltr, ytr).score(Xlte, yte)
print(f"leakage: clean test accuracy {clean:.3f}, with leaked feature {leaky:.3f} (unattainable in production)")

# Group leakage: repeated measurements per patient
patients = np.repeat(np.arange(300), 10)
signature = rng.normal(size=(300, 3))[patients] + rng.normal(0, 0.1, (3000, 3))   # each patient has a fingerprint
yp = rng.integers(0, 2, 300)[patients]                    # label is per patient and unrelated to features
from sklearn.neighbors import KNeighborsClassifier       # noqa: E402
rand_cv = cross_val_score(KNeighborsClassifier(1), signature, yp, cv=KFold(5, shuffle=True, random_state=0)).mean()
group_cv = cross_val_score(KNeighborsClassifier(1), signature, yp, cv=GroupKFold(5), groups=patients).mean()
print(f"group leakage: random-split CV {rand_cv:.3f} vs group-split CV {group_cv:.3f} (labels are pure noise!)")

# ---------------------------------------------------------------------------
# 3. Splits figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 3.6), sharex=True)
N = 40
colors = {0: BLUE, 1: ORANGE, 2: PURPLE}
idx = rng.permutation(N)
random_split = np.zeros(N, int); random_split[idx[28:34]] = 1; random_split[idx[34:]] = 2
time_split = np.r_[np.zeros(28, int), np.ones(6, int), 2 * np.ones(6, int)]
groups = np.repeat(np.arange(8), 5)
gperm = rng.permutation(8)
group_split = np.select([np.isin(groups, gperm[6:7]), np.isin(groups, gperm[7:])], [1, 2], 0)
for ax, s, title in [(axes[0], random_split, "Random: i.i.d. rows"), (axes[1], time_split, "Time: train on the past, test on the future"), (axes[2], group_split, "Group: all rows of a user/patient stay together")]:
    for k in range(N):
        ax.add_patch(plt.Rectangle((k, 0), 0.9, 1, color=colors[s[k]]))
    if ax is axes[2]:
        for gb in range(1, 8):
            ax.axvline(gb * 5 - 0.05, color="white", lw=2)
    ax.set(xlim=(0, N), ylim=(0, 1), yticks=[])
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(False)
axes[2].set_xlabel("rows (time order for the middle panel)")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in range(3)]
fig.legend(handles, ["train", "validation", "test"], loc="upper right", frameon=False, fontsize=8, ncol=3)
fig.subplots_adjust(hspace=0.9)
save(fig, "ml-workflow")

# ---------------------------------------------------------------------------
# 4. Under/overfitting and validation curve
# ---------------------------------------------------------------------------
x = np.sort(rng.uniform(0, 1, 30))
yy = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 30)
xv = rng.uniform(0, 1, 200)
yv = np.sin(2 * np.pi * xv) + rng.normal(0, 0.25, 200)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
grid = np.linspace(0, 1, 300)
for ax, d, label in [(axes[0], 1, "degree 1: underfit"), (axes[1], 4, "degree 4: good fit"), (axes[2], 15, "degree 15: overfit")]:
    mdl = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], yy)
    tr = np.mean((mdl.predict(x[:, None]) - yy) ** 2)
    va = np.mean((mdl.predict(xv[:, None]) - yv) ** 2)
    ax.scatter(x, yy, s=14, color=BLUE)
    ax.plot(grid, np.sin(2 * np.pi * grid), color=GRAY, ls="--", lw=1)
    ax.plot(grid, mdl.predict(grid[:, None]), color=ORANGE, lw=2)
    ax.set(title=f"{label}\ntrain MSE {tr:.3f} · val MSE {va:.2f}", ylim=(-2, 2), xlabel="x")
    ax.title.set_fontsize(9)
    print(f"degree {d:2d}: train MSE {tr:.3f}, validation MSE {va:.3f}")
degrees = range(1, 16)
tr_err, va_err = [], []
for d in degrees:
    mdl = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], yy)
    tr_err.append(np.mean((mdl.predict(x[:, None]) - yy) ** 2))
    va_err.append(np.mean((mdl.predict(xv[:, None]) - yv) ** 2))
ax = axes[3]
ax.plot(list(degrees), tr_err, "-o", ms=3, color=BLUE, label="training error")
ax.plot(list(degrees), va_err, "-o", ms=3, color=ORANGE, label="validation error")
ax.set(title="Validation curve", xlabel="polynomial degree (model complexity)", ylabel="MSE", yscale="log")
ax.title.set_fontsize(9)
ax.legend(frameon=False, fontsize=8)
best = int(np.argmin(va_err)) + 1
print(f"best degree by validation: {best}")
fig.subplots_adjust(wspace=0.32)
save(fig, "overfitting-and-bias-variance")
