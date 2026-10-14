# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/17-imbalanced-data
"""Imbalanced classification from scratch: resampling (random over/under, SMOTE), class weights,
threshold tuning on a validation set, and metrics that don't lie (precision, recall, F1, MCC, AUPRC).

All data is SYNTHETIC (two overlapping Gaussian blobs, 5% positives). Deterministic.
Run from guide/:  uv run code/24-applied-ml/imbalanced_data.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, INK, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


# ---------- data ----------
def make_data(n_neg, n_pos, rng):
    X_neg = rng.normal(0.0, 1.0, size=(n_neg, 2))
    X_pos = rng.normal(1.3, 0.8, size=(n_pos, 2))  # overlaps the negatives, so the task is not trivial
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------- metrics ----------
def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def metrics(y, pred):
    tp, fp, fn, tn = confusion(y, pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def average_precision(y, scores):
    order = np.argsort(-scores)             # rank by score, highest first
    hits = y[order]
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(np.sum(precision_at_k * hits) / hits.sum())  # mean precision at each true positive


# ---------- resampling ----------
def random_oversample(X, y, rng):
    pos = np.where(y == 1)[0]
    extra = rng.choice(pos, size=int((y == 0).sum()) - len(pos), replace=True)  # duplicates
    keep = np.concatenate([np.arange(len(y)), extra])
    return X[keep], y[keep]


def random_undersample(X, y, rng):
    pos = np.where(y == 1)[0]
    neg = rng.choice(np.where(y == 0)[0], size=len(pos), replace=False)  # discards most negatives
    keep = np.concatenate([pos, neg])
    return X[keep], y[keep]


def smote(X_min, n_new, k, rng):
    d = np.linalg.norm(X_min[:, None, :] - X_min[None, :, :], axis=2)  # [m, m] pairwise distances
    np.fill_diagonal(d, np.inf)                                        # a point is not its own neighbor
    neighbors = np.argsort(d, axis=1)[:, :k]                           # [m, k] k nearest minority points
    base = rng.integers(0, len(X_min), size=n_new)                     # which real point to start from
    pick = neighbors[base, rng.integers(0, k, size=n_new)]             # one of its neighbors
    t = rng.random((n_new, 1))                                         # position along the segment
    return X_min[base] + t * (X_min[pick] - X_min[base])


# ---------- model ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_logreg(X, y, sample_weight=None, lr=0.5, epochs=2000):
    w_s = np.ones(len(y)) if sample_weight is None else sample_weight
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        err = (sigmoid(X @ w + b) - y) * w_s           # each row's gradient scaled by its weight
        w -= lr * (X.T @ err) / w_s.sum()
        b -= lr * err.sum() / w_s.sum()
    return w, b


def balanced_weights(y):
    n, counts = len(y), np.bincount(y.astype(int))     # same rule as sklearn class_weight="balanced"
    per_class = n / (len(counts) * counts)
    return per_class[y.astype(int)], per_class


def best_threshold(y, probs):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = np.array([metrics(y, (probs >= t).astype(int))["f1"] for t in grid])
    return float(grid[f1s.argmax()]), grid, f1s


def main():
    rng = np.random.default_rng(0)
    X, y = make_data(n_neg=3800, n_pos=200, rng=rng)  # 5% positive
    X_tr, y_tr = X[:2400], y[:2400]
    X_va, y_va = X[2400:3200], y[2400:3200]
    X_te, y_te = X[3200:], y[3200:]
    print(f"positives: train {int(y_tr.sum())}/{len(y_tr)}, val {int(y_va.sum())}/{len(y_va)}, "
          f"test {int(y_te.sum())}/{len(y_te)}  (synthetic)")

    # The accuracy trap: predict 0 for everyone.
    always_neg = np.zeros_like(y_te)
    acc = float(np.mean(always_neg == y_te))
    print(f"always-negative on test: accuracy {acc:.3f}, recall 0.000, F1 0.000, MCC 0.000")

    # Worked example numbers for the lesson.
    _, per_class = balanced_weights(y_tr)
    n_pos_tr = int(y_tr.sum())
    print(f"class weights: n={len(y_tr)}, n_pos={n_pos_tr}, "
          f"w_neg={per_class[0]:.4f}, w_pos={per_class[1]:.4f}, ratio={per_class[1] / per_class[0]:.2f}")
    a, nb = np.array([1.0, 2.0]), np.array([1.5, 2.5])
    print(f"SMOTE step: x={a.tolist()}, neighbor={nb.tolist()}, t=0.4 -> {(a + 0.4 * (nb - a)).tolist()}")

    # Train each strategy on its own version of the training set.
    X_min = X_tr[y_tr == 1]
    X_sm = smote(X_min, n_new=int((y_tr == 0).sum()) - len(X_min), k=5, rng=rng)
    sets = {
        "baseline": (X_tr, y_tr, None),
        "oversample": (*random_oversample(X_tr, y_tr, rng), None),
        "undersample": (*random_undersample(X_tr, y_tr, rng), None),
        "SMOTE": (np.vstack([X_tr, X_sm]), np.concatenate([y_tr, np.ones(len(X_sm))]), None),
        "class weights": (X_tr, y_tr, balanced_weights(y_tr)[0]),
    }
    models = {name: fit_logreg(Xs, ys, ws) for name, (Xs, ys, ws) in sets.items()}

    # Threshold tuning on the VALIDATION set (never on test).
    tuned = {}
    for name in ["baseline", "class weights"]:
        w, b = models[name]
        tuned[name] = best_threshold(y_va, sigmoid(X_va @ w + b))
        print(f"best F1 threshold for {name} (tuned on validation): {tuned[name][0]:.2f}")
    t_star, grid, f1s = tuned["baseline"]

    print(f"\n{'strategy':<24}{'thresh':>7}{'prec':>7}{'recall':>8}{'F1':>7}{'MCC':>7}{'AUPRC':>7}")
    rows = {}
    runs = [(n, n, 0.5) for n in models] + [(f"{n} + tuned t", n, tuned[n][0]) for n in tuned]
    for name, model, t in runs:
        w, b = models[model]
        probs = sigmoid(X_te @ w + b)
        m = metrics(y_te, (probs >= t).astype(int))
        m["auprc"] = average_precision(y_te, probs)
        rows[name] = m
        print(f"{name:<24}{t:>7.2f}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}"
              f"{m['mcc']:>7.3f}{m['auprc']:>7.3f}")
    print(f"AUPRC of a random scorer = positive rate = {y_te.mean():.3f}")

    # Worked MCC example on the class-weights row.
    w, b = models["class weights"]
    tp, fp, fn, tn = confusion(y_te, (sigmoid(X_te @ w + b) >= 0.5).astype(int))
    print(f"\nclass-weights confusion on test: TP={tp} FP={fp} FN={fn} TN={tn}")

    # Prior correction: turn a balanced-trained probability back into a real-world one.
    pi = n_pos_tr / len(y_tr)
    p_bal = 0.6
    odds = p_bal / (1 - p_bal) * pi / (1 - pi)
    print(f"prior correction: p_balanced={p_bal}, pi={pi:.4f} -> p_real={odds / (1 + odds):.4f}")

    # Self-checks against scikit-learn.
    sk = LogisticRegression(class_weight="balanced", C=1e6).fit(X_tr, y_tr)
    sk_pred = sk.predict(X_te)
    ours_pred = (sigmoid(X_te @ w + b) >= 0.5).astype(int)
    agree = float(np.mean(sk_pred == ours_pred))
    assert agree > 0.99, agree
    assert abs(average_precision(y_te, sigmoid(X_te @ w + b))
               - average_precision_score(y_te, sigmoid(X_te @ w + b))) < 1e-9
    assert abs(rows["class weights"]["f1"] - f1_score(y_te, ours_pred)) < 1e-9
    assert abs(rows["class weights"]["mcc"] - matthews_corrcoef(y_te, ours_pred)) < 1e-9
    print(f"checks passed: matches sklearn class_weight='balanced' on {agree:.1%} of test rows")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"wspace": 0.55})
    ax = axes[0]
    ax.scatter(*X_tr[y_tr == 0].T, s=5, color=GRAY, alpha=0.4, label="negative (train)")
    ax.scatter(*X_sm[:300].T, s=8, color=TEAL, alpha=0.6, marker="x", label="SMOTE synthetic (300 shown)")
    ax.scatter(*X_min.T, s=14, color=ORANGE, label="positive (train)")
    xs = np.linspace(-3.5, 4.5, 50)
    for name, color in [("baseline", BLUE), ("class weights", PURPLE)]:
        w, b = models[name]
        ax.plot(xs, -(w[0] * xs + b) / w[1], color=color, lw=2, label=f"{name}: p = 0.5")
    ax.set(xlim=(-3.5, 4.5), ylim=(-3.5, 4.5), xlabel="feature 1", ylabel="feature 2",
           title="Re-weighting moves the boundary")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ax.plot(grid, f1s, color=BLUE)
    ax.axvline(0.5, color=GRAY, ls="--", lw=1)
    ax.axvline(t_star, color=ORANGE, lw=1.5)
    ax.text(t_star + 0.02, f1s.max() * 0.25, f"best t = {t_star:.2f}", color=ORANGE)
    ax.text(0.52, f1s.max() * 0.1, "default 0.5", color=GRAY)
    ax.set(xlabel="threshold t (baseline model)", ylabel="F1 on validation",
           title="Tune the threshold on validation")

    ax = axes[2]
    names = list(rows)
    pos = np.arange(len(names))
    ax.barh(pos - 0.2, [rows[n]["recall"] for n in names], height=0.4, color=TEAL, label="recall")
    ax.barh(pos + 0.2, [rows[n]["precision"] for n in names], height=0.4, color=PURPLE, label="precision")
    for i, n in enumerate(names):
        ax.text(max(rows[n]["recall"], rows[n]["precision"]) + 0.02, i, f"F1 {rows[n]['f1']:.2f}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(pos, names)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.25), xlabel="test set", title="Same ranking, different trade-off")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "imbalanced-data")


if __name__ == "__main__":
    main()
