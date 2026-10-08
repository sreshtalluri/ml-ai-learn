"""Logistic regression worked example, softmax/cross-entropy, threshold metrics, ROC/PR, and calibration.

Run from guide/:  uv run code/04-classification/classification.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. Logistic regression by hand
# ---------------------------------------------------------------------------
w, b, x = 1.5, -4.0, 3.0
z = w * x + b
p = sigmoid(z)
print(f"z = {w}*{x} + ({b}) = {z};  p = sigmoid(z) = {p:.4f}")
print(f"BCE if y=1: {-np.log(p):.4f};  if y=0: {-np.log(1 - p):.4f}")
print(f"gradient dL/dw for y=1: (p - y) x = {(p - 1) * x:.4f}")
logits = np.array([2.0, 1.0, 0.1])
sm = np.exp(logits) / np.exp(logits).sum()
print(f"softmax({logits}) = {sm.round(4)}; CCE (true class 0) = {-np.log(sm[0]):.4f}")

# ---------------------------------------------------------------------------
# 2. Ten scored examples: confusion matrix at two thresholds
# ---------------------------------------------------------------------------
scores = np.array([0.95, 0.85, 0.80, 0.70, 0.55, 0.45, 0.40, 0.30, 0.20, 0.10])
labels = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
for t in (0.5, 0.3):
    pred = (scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    print(f"t={t}: TP={tp} FP={fp} FN={fn} TN={tn} precision={precision_score(labels, pred):.3f} "
          f"recall={recall_score(labels, pred):.3f} F1={f1_score(labels, pred):.3f} accuracy={(tp + tn) / 10:.2f}")

# ---------------------------------------------------------------------------
# 3. SYNTHETIC imbalanced dataset (5% positives)
# ---------------------------------------------------------------------------
X, y = make_classification(n_samples=6000, n_features=8, n_informative=4, weights=[0.95], flip_y=0.02, random_state=3)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
s = clf.predict_proba(Xte)[:, 1]
print(f"\nImbalanced test set: {yte.mean():.3f} positive. ROC AUC={roc_auc_score(yte, s):.3f}  PR AUC (AP)={average_precision_score(yte, s):.3f}")
print(f"Always-negative accuracy: {1 - yte.mean():.3f};  model accuracy at 0.5: {((s >= 0.5) == yte).mean():.3f}, recall at 0.5: {recall_score(yte, s >= 0.5):.3f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
ax = axes[0]
zz = np.linspace(-6, 6, 200)
ax.plot(zz, sigmoid(zz), color=BLUE, lw=2)
ax.axhline(0.5, color=GRAY, ls="--", lw=1)
ax.plot([z], [p], "o", color=ORANGE)
ax.annotate(f"z = {z}, p = {p:.3f}", (z, p), xytext=(1.2, 0.35), fontsize=9, arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.set(title="Sigmoid turns a logit into a probability", xlabel="logit z = w·x + b", ylabel="p(y = 1 | x)")
ax.title.set_fontsize(10)

ax = axes[1]
ax.hist(s[yte == 0], bins=40, color=BLUE, alpha=0.5, density=True, label="negatives")
ax.hist(s[yte == 1], bins=40, color=ORANGE, alpha=0.6, density=True, label="positives")
ax.axvline(0.5, color="black", lw=1.5)
ax.set(title="Scores by true class (5% positive)", xlabel="predicted probability", ylabel="density", yscale="log")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
fpr, tpr, _ = roc_curve(yte, s)
prec, rec, _ = precision_recall_curve(yte, s)
ax.plot(fpr, tpr, color=PURPLE, lw=2, label=f"ROC (AUC {roc_auc_score(yte, s):.2f})")
ax.plot(rec, prec, color=TEAL, lw=2, label=f"PR (AP {average_precision_score(yte, s):.2f})")
ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
ax.axhline(yte.mean(), color=TEAL, ls=":", lw=1)
ax.set(title="ROC vs precision-recall", xlabel="FPR (ROC) / recall (PR)", ylabel="TPR (ROC) / precision (PR)")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)

ax = axes[3]
frac, mean_pred = calibration_curve(yte, s, n_bins=8, strategy="quantile")
ax.plot(mean_pred, frac, "-o", color=ORANGE, label="logistic regression")
ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="perfect calibration")
ax.set(title="Calibration (reliability diagram)", xlabel="mean predicted probability", ylabel="observed positive rate")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
fig.subplots_adjust(wspace=0.35)
save(fig, "classification-metrics")

# Logistic regression figure: 2D boundary and probability contours
X2, y2 = make_classification(n_samples=300, n_features=2, n_redundant=0, n_clusters_per_class=1, class_sep=1.2, random_state=8)
m2 = LogisticRegression().fit(X2, y2)
gx, gy = np.meshgrid(np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 200), np.linspace(X2[:, 1].min() - 1, X2[:, 1].max() + 1, 200))
pp = m2.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
cs = ax.contourf(gx, gy, pp, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
ax.contour(gx, gy, pp, levels=[0.5], colors="black", linewidths=1.5)
ax.scatter(*X2[y2 == 0].T, s=10, color=BLUE)
ax.scatter(*X2[y2 == 1].T, s=10, color=ORANGE, marker="s")
fig.colorbar(cs, ax=ax, label="p(y = 1)")
ax.set(title="Logistic regression: linear boundary (black) at p = 0.5", xlabel="x₁", ylabel="x₂")
ax.title.set_fontsize(10)
ax = axes[1]
pg = np.linspace(0.01, 0.99, 200)
ax.plot(pg, -np.log(pg), color=ORANGE, lw=2, label="y = 1: −log p")
ax.plot(pg, -np.log(1 - pg), color=BLUE, lw=2, label="y = 0: −log(1 − p)")
ax.set(title="Binary cross-entropy punishes confident mistakes", xlabel="predicted p", ylabel="loss", ylim=(0, 4.6))
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
save(fig, "logistic-regression")
