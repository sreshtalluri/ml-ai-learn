# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/08-feature-engineering
"""Feature engineering from scratch: scaling, log transform, encodings, imputation, and filter selection.

The headline experiment shows target-encoding leakage: a pure-noise ID column looks predictive when it is
target-encoded on the same rows the model trains on, and stops looking predictive when encoded out of fold.
All data is SYNTHETIC and seeded.
Run from guide/:  uv run code/24-applied-ml/feature_engineering.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402


# ---------- numerical transforms ----------
def standardize(x):
    """z = (x - mean) / std, using population std. Fit on train, reuse mean/std on test."""
    x = np.asarray(x, dtype=float)
    mean, std = x.mean(), x.std()
    return (x - mean) / (std if std > 0 else 1.0), mean, std


def min_max(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def log1p(x):
    """log(1 + x): compresses a right tail, keeps 0 at 0. Only for x > -1."""
    return np.log1p(np.asarray(x, dtype=float))


# ---------- categorical encodings ----------
def one_hot(values):
    cats = sorted(set(values))
    idx = {c: i for i, c in enumerate(cats)}
    out = np.zeros((len(values), len(cats)), dtype=int)
    for r, v in enumerate(values):
        out[r, idx[v]] = 1
    return out, cats


def target_encode_fit(cats, y, m=10.0):
    """Smoothed mean: enc(c) = (n_c * mean_c + m * global_mean) / (n_c + m)."""
    y = np.asarray(y, dtype=float)
    g = y.mean()
    table = {}
    for c in set(cats):
        mask = np.array([v == c for v in cats])
        n_c, mean_c = mask.sum(), y[mask].mean()
        table[c] = (n_c * mean_c + m * g) / (n_c + m)
    return table, g


def target_encode_oof(cats, y, m=10.0, k=5, seed=0):
    """Out-of-fold: each row is encoded with a table fit on the OTHER folds, so its own label never leaks in."""
    cats, y = np.asarray(cats), np.asarray(y, dtype=float)
    folds = np.random.default_rng(seed).permutation(len(y)) % k
    out = np.empty(len(y))
    for f in range(k):
        table, g = target_encode_fit(list(cats[folds != f]), y[folds != f], m)
        out[folds == f] = [table.get(c, g) for c in cats[folds == f]]
    return out


# ---------- missing values ----------
def impute_median_with_indicator(x):
    """Fill NaN with the median of present values, and return a was_missing column."""
    x = np.asarray(x, dtype=float)
    miss = np.isnan(x)
    med = np.median(x[~miss])
    return np.where(miss, med, x), miss.astype(int), med


# ---------- filter selection ----------
def mutual_information(feature, target, n_bins=10):
    """MI between an equal-width-binned feature and a discrete target, in nats."""
    f = np.asarray(feature, dtype=float)
    t = np.asarray(target)
    edges = np.linspace(f.min(), f.max(), n_bins + 1)
    b = np.clip(np.digitize(f, edges[1:-1]), 0, n_bins - 1)
    mi = 0.0
    for bi in np.unique(b):
        for ti in np.unique(t):
            p_joint = np.mean((b == bi) & (t == ti))
            if p_joint > 0:
                mi += p_joint * math.log(p_joint / (np.mean(b == bi) * np.mean(t == ti)))
    return mi


def drop_correlated(X, threshold=0.9):
    """Greedy: keep column i, drop any later column j with |r_ij| >= threshold."""
    r = np.corrcoef(X, rowvar=False)
    keep = []
    for j in range(X.shape[1]):
        if all(abs(r[i, j]) < threshold for i in keep):
            keep.append(j)
    return keep


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


if __name__ == "__main__":
    print("=== Worked example 1: standardization ===")
    x = [2, 4, 4, 4, 5, 5, 7, 9]
    z, mean, std = standardize(x)
    print(f"x = {x}")
    print(f"mean = {mean:.1f}, std = {std:.1f}, z = {np.round(z, 2).tolist()}")
    assert mean == 5 and std == 2

    print("\n=== Worked example 2: log1p on a skewed feature ===")
    incomes = [20_000, 40_000, 60_000, 1_000_000]
    print(f"income            = {incomes}")
    print(f"log1p(income)     = {np.round(log1p(incomes), 2).tolist()}")
    print(f"max/min raw = {max(incomes) / min(incomes):.0f}x, after log1p gap = "
          f"{log1p(1_000_000) - log1p(20_000):.2f} nats")

    print("\n=== Worked example 3: smoothed target encoding (m = 10) ===")
    cats = ["city"] * 6 + ["suburb"] * 2 + ["rural"] * 2
    y = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    table, g = target_encode_fit(cats, y, m=10)
    print(f"global mean = {g:.2f}")
    for c in ["city", "suburb", "rural"]:
        print(f"  {c:7s} n={cats.count(c)}  raw mean={np.mean([t for v, t in zip(cats, y) if v == c]):.2f}"
              f"  encoded={table[c]:.4f}")
    assert abs(table["rural"] - (2 * 1.0 + 10 * 0.3) / 12) < 1e-12

    print("\n=== Worked example 4: median imputation + indicator ===")
    sqft = [800, np.nan, 1200, 1500, np.nan, 3000]
    filled, flag, med = impute_median_with_indicator(sqft)
    print(f"median of present = {med:.0f}, filled = {filled.tolist()}, was_missing = {flag.tolist()}")

    print("\n=== Leakage experiment: target-encoding a pure-noise ID column (SYNTHETIC) ===")
    rng = np.random.default_rng(42)
    n_train, n_test, n_ids = 2000, 2000, 1000
    signal_tr, signal_te = rng.normal(size=n_train), rng.normal(size=n_test)
    y_tr = (signal_tr + rng.normal(scale=1.5, size=n_train) > 0).astype(int)
    y_te = (signal_te + rng.normal(scale=1.5, size=n_test) > 0).astype(int)
    ids_tr = rng.integers(0, n_ids, n_train).astype(str)   # ~2 rows per ID: pure noise
    ids_te = rng.integers(0, n_ids, n_test).astype(str)

    results = {}
    base = LogisticRegression().fit(signal_tr[:, None], y_tr)
    results["signal only"] = (accuracy(base, signal_tr[:, None], y_tr), accuracy(base, signal_te[:, None], y_te))

    for name, enc_tr in [("naive TE", None), ("out-of-fold TE", "oof")]:
        table, g = target_encode_fit(list(ids_tr), y_tr, m=1.0)
        te_tr = target_encode_oof(ids_tr, y_tr, m=1.0) if enc_tr else np.array([table[c] for c in ids_tr])
        te_te = np.array([table.get(c, g) for c in ids_te])
        Xtr, Xte = np.c_[signal_tr, te_tr], np.c_[signal_te, te_te]
        model = LogisticRegression().fit(Xtr, y_tr)
        results[name] = (accuracy(model, Xtr, y_tr), accuracy(model, Xte, y_te))
        print(f"  {name:15s} weight on ID feature = {model.coef_[0][1]:+.2f}")
    for name, (a_tr, a_te) in results.items():
        print(f"  {name:15s} train acc = {a_tr:.3f}   test acc = {a_te:.3f}")
    assert results["naive TE"][0] > results["out-of-fold TE"][0] + 0.05   # naive looks better on train
    assert results["naive TE"][1] < results["out-of-fold TE"][1]          # ...and is worse on test

    print("\n=== Filter selection on a SYNTHETIC housing table ===")
    n = 1000
    sqft_h = rng.uniform(500, 5000, n)
    sqft_m2 = sqft_h * 0.0929 + rng.normal(0, 1, n)          # near-duplicate of sqft
    age = rng.uniform(0, 50, n)
    noise = rng.normal(size=n)
    constant_ish = np.where(rng.random(n) < 0.995, 1.0, 0.0)  # almost never varies
    price = 50 * sqft_h - 1000 * age + rng.normal(0, 20000, n)
    expensive = (price > np.median(price)).astype(int)
    X = np.c_[sqft_h, sqft_m2, age, noise, constant_ish]
    names = ["sqft", "sqft_m2", "age", "noise", "constant_ish"]
    var = [min_max(X[:, j]).var() for j in range(X.shape[1])]
    keep = drop_correlated(X, 0.9)
    mis = [mutual_information(X[:, j], expensive) for j in range(X.shape[1])]
    for j, nm in enumerate(names):
        why = ("drop: variance < 0.01" if var[j] < 0.01 else
               "drop: |r| >= 0.9 with an earlier column" if j not in keep else "keep")
        print(f"  {nm:12s} var(min-max) = {var[j]:.4f}   MI = {mis[j]:.4f} nats   {why}")

    print("\n=== Library version: ColumnTransformer + TargetEncoder (cross-fitted) ===")
    Xmix = np.empty((n_train, 3), dtype=object)
    Xmix[:, 0], Xmix[:, 1], Xmix[:, 2] = signal_tr, ids_tr, rng.choice(["a", "b", "c"], n_train)
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), [0]),
        ("ids", TargetEncoder(target_type="binary"), [1]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [2]),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression())]).fit(Xmix, y_tr)
    Xmix_te = np.empty((n_test, 3), dtype=object)
    Xmix_te[:, 0], Xmix_te[:, 1], Xmix_te[:, 2] = signal_te, ids_te, rng.choice(["a", "b", "c"], n_test)
    print(f"  pipeline test acc = {pipe.score(Xmix_te, y_te):.3f}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    skewed = rng.lognormal(mean=10.5, sigma=0.9, size=3000)
    axes[0].hist(skewed / 1000, bins=60, color=BLUE)
    axes[0].set(title="Raw income (synthetic)", xlabel="income (thousands)", ylabel="count")
    axes[1].hist(log1p(skewed), bins=60, color=TEAL)
    axes[1].set(title="After log1p", xlabel="log(1 + income)")
    labels = list(results)
    xs = np.arange(len(labels))
    axes[2].bar(xs - 0.2, [results[k][0] for k in labels], 0.4, color=GRAY, label="train")
    axes[2].bar(xs + 0.2, [results[k][1] for k in labels], 0.4, color=ORANGE, label="test")
    for i, k in enumerate(labels):
        axes[2].text(i - 0.2, results[k][0] + 0.01, f"{results[k][0]:.2f}", ha="center", fontsize=8)
        axes[2].text(i + 0.2, results[k][1] + 0.01, f"{results[k][1]:.2f}", ha="center", fontsize=8)
    axes[2].set_xticks(xs, labels)
    axes[2].set(ylim=(0.5, 1.0), ylabel="accuracy", title="Noise ID column, target-encoded")
    axes[2].legend(loc="upper right", frameon=False)
    save(fig, "feature-engineering")
