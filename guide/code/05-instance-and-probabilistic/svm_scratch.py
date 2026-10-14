# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/05-support-vector-machines
"""Linear SVM from scratch (NumPy subgradient descent on the hinge loss) plus an RBF kernel, checked against scikit-learn.

Run from guide/:  uv run code/05-instance-and-probabilistic/svm_scratch.py
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


# --- Chunk 1: hinge loss and the soft-margin objective ------------------------
def hinge(X, y, w, b):
    margins = y * (X @ w + b)                  # y_i f(x_i): >= 1 means safely outside the street
    return np.maximum(0.0, 1.0 - margins)      # [n], zero for points beyond the margin


def objective(X, y, w, b, lam):                # lambda/2 ||w||^2 + mean hinge
    return 0.5 * lam * w @ w + hinge(X, y, w, b).mean()


w0, b0 = np.array([3.0, 4.0]), -2.0            # the lesson's worked example
pts = np.array([[1.0, 0.5], [0.2, 0.3]])
print(f"worked example: margin width {2 / np.linalg.norm(w0):.2f}, "
      f"hinge losses {hinge(pts, np.array([1, 1]), w0, b0)}")


# --- Chunk 2: one subgradient step --------------------------------------------
def step(X, y, w, b, lam, lr):
    active = y * (X @ w + b) < 1               # only margin violators push on w and b
    grad_w = lam * w - (y[active, None] * X[active]).sum(axis=0) / len(y)
    grad_b = -y[active].sum() / len(y)
    return w - lr * grad_w, b - lr * grad_b


# --- Chunk 3: train by repeating the step --------------------------------------
def fit_linear_svm(X, y, lam=0.01, lr=0.5, epochs=3000):
    w, b = np.zeros(X.shape[1]), 0.0
    for t in range(epochs):
        w, b = step(X, y, w, b, lam, lr / np.sqrt(1 + t))   # decaying step for a non-smooth loss
    return w, b


# SYNTHETIC: two overlapping blobs, labels in {-1, +1}
X, y01 = make_blobs(n_samples=200, centers=[[-1.5, 0], [1.5, 0.5]], cluster_std=1.0, random_state=6)
y = 2 * y01 - 1
lam = 0.01
w, b = fit_linear_svm(X, y, lam)
print(f"\nscratch: w={w.round(3)} b={b:.3f}  objective {objective(X, y, w, b, lam):.4f}  "
      f"train acc {np.mean(np.sign(X @ w + b) == y):.3f}")

# Use the library: same objective with C = 1 / (lambda * n)
svc = SVC(kernel="linear", C=1 / (lam * len(y))).fit(X, y)
w_l, b_l = svc.coef_[0], svc.intercept_[0]
print(f"sklearn: w={w_l.round(3)} b={b_l:.3f}  objective {objective(X, y, w_l, b_l, lam):.4f}  "
      f"train acc {svc.score(X, y):.3f}  support vectors {len(svc.support_)}")


# --- Chunk 4: the RBF kernel matrix, similarity without building features -------
def rbf_kernel(A, B, gamma=0.5):
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T   # ||a - b||^2 for all pairs
    return np.exp(-gamma * np.maximum(sq, 0.0))


K = rbf_kernel(np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]), np.array([[0.0, 0.0]]))
print(f"\nRBF similarity to (0,0), gamma=0.5: {K.ravel().round(4)}")

assert abs(objective(X, y, w, b, lam) - objective(X, y, w_l, b_l, lam)) < 1e-3
