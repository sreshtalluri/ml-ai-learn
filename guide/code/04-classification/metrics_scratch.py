# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/09-model-evaluation
"""Confusion matrix, precision/recall/F1, ROC curve and AUC, average precision, and a cost-based
threshold, from scratch, checked against scikit-learn. Scores are SYNTHETIC.

Run from guide/:  uv run code/04-classification/metrics_scratch.py
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0         # of flagged, how many real
    recall = tp / (tp + fn) if tp + fn else 0.0            # of real, how many flagged
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_points(y, s):
    """Sweep the threshold from high to low; each step lets one more example through as positive."""
    # ponytail: assumes no tied scores; sklearn merges tied scores into one ROC point
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tps = np.cumsum(y)                                     # true positives above each cut
    fps = np.cumsum(1 - y)
    tpr = np.r_[0, tps / y.sum()]
    fpr = np.r_[0, fps / (len(y) - y.sum())]
    return fpr, tpr


def auc_trapezoid(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def auc_ranking(y, s):
    """AUC = P(random positive scores higher than random negative), ties count half."""
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def average_precision(y, s):
    """Mean of precision at the rank of each positive."""
    order = np.argsort(-s, kind="stable")
    y = y[order]
    hits = np.cumsum(y)
    precision_at_k = hits / np.arange(1, len(y) + 1)
    return float(np.sum(precision_at_k * y) / y.sum())


if __name__ == "__main__":
    # The lesson's 10-example worked case
    s10 = np.array([0.95, 0.85, 0.80, 0.70, 0.55, 0.45, 0.40, 0.30, 0.20, 0.10])
    y10 = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    for t in (0.5, 0.3):
        tp, fp, fn, tn = confusion(y10, (s10 >= t).astype(int))
        p, r, f = prf(tp, fp, fn)
        print(f"t={t}: TP={tp} FP={fp} FN={fn} TN={tn}  precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
    fpr, tpr = roc_points(y10, s10)
    print(f"AUC: trapezoid {auc_trapezoid(fpr, tpr):.3f}, ranking {auc_ranking(y10, s10):.3f}, AP {average_precision(y10, s10):.3f}")

    # Imbalanced SYNTHETIC scores: 5% positives
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(int)
    s = np.clip(rng.normal(0.25 + 0.35 * y, 0.15), 0, 1)
    fpr, tpr = roc_points(y, s)
    print(f"imbalanced: ROC AUC {auc_trapezoid(fpr, tpr):.3f}, AP {average_precision(y, s):.3f}, base rate {y.mean():.3f}")

    c_fp, c_fn = 1, 10                                     # a missed positive costs 10x a false alarm
    ts = np.linspace(0, 1, 101)
    costs = []
    for t in ts:
        tp, fp, fn, tn = confusion(y, (s >= t).astype(int))
        costs.append(c_fp * fp + c_fn * fn)
    t_best = ts[int(np.argmin(costs))]
    print(f"cost-minimizing threshold {t_best:.2f} (cost {min(costs)}), cost at 0.5: {costs[50]}")

    assert np.isclose(auc_trapezoid(*roc_points(y, s)), roc_auc_score(y, s))
    assert np.isclose(auc_ranking(y, s), roc_auc_score(y, s))
    assert np.isclose(average_precision(y, s), average_precision_score(y, s))
    assert np.isclose(prf(*confusion(y, (s >= 0.5).astype(int))[:3])[2], f1_score(y, s >= 0.5))
    print("checks pass: all metrics match scikit-learn")
