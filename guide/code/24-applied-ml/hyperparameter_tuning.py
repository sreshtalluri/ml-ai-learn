# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/12-hyperparameter-tuning
"""Hyperparameter search from scratch: grid vs random, successive halving, and Bayesian optimization (GP + EI).

Parts 1 and 2 use a SYNTHETIC score surface and SYNTHETIC learning curves so the comparison is exact and fast.
Parts 3 and 4 tune a real SVC with cross-validation on a SYNTHETIC classification dataset.
Run from guide/:  uv run code/24-applied-ml/hyperparameter_tuning.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402


# ---------- Part 1: a SYNTHETIC score surface where only the learning rate matters much ----------
def score_surface(log_lr, depth):
    """Validation score: peaks at lr = 10^-2 = 0.01; depth barely matters. Max 0.90."""
    return 0.90 - 0.08 * (log_lr + 2.0) ** 2 - 0.002 * (depth - 6) ** 2


def grid_points(n_per_axis):
    lrs = np.linspace(-3, 0, n_per_axis)            # log10 learning rate in [1e-3, 1]
    depths = np.linspace(2, 10, n_per_axis)
    return np.array([(a, b) for a in lrs for b in depths])


def random_points(n, rng):
    return np.c_[rng.uniform(-3, 0, n), rng.uniform(2, 10, n)]


# ---------- Part 2: successive halving on SYNTHETIC learning curves ----------
def curve(final, budget):
    """Score after `budget` epochs: approaches `final` like 1/sqrt(budget)."""
    return final - 0.15 / math.sqrt(budget)


def successive_halving(finals, min_budget=1, max_budget=9, eta=3, rng=None):
    """Run all configs at min_budget, keep the top 1/eta, multiply budget by eta, until max_budget."""
    alive = np.arange(len(finals))
    budget, cost, rounds = min_budget, 0, []
    while True:
        scores = np.array([curve(finals[i], budget) + rng.normal(0, 0.01) for i in alive])
        cost += budget * len(alive)
        rounds.append((budget, alive.copy(), scores))
        if budget >= max_budget:
            return alive[np.argmax(scores)], cost, rounds
        alive = alive[np.argsort(-scores)[: max(1, len(alive) // eta)]]
        budget *= eta


# ---------- Part 3: Bayesian optimization, Gaussian-process surrogate + expected improvement ----------
def rbf(a, b, length=0.8):
    """Squared-exponential kernel between rows of a [n, d] and b [m, d]. Returns [n, m]."""
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / length**2)


def gp_posterior(x_obs, y_obs, x_new, noise=0.05):
    """GP with zero mean on standardized y. Returns predictive mean and std at x_new."""
    mu_y, sd_y = y_obs.mean(), y_obs.std() + 1e-9
    y = (y_obs - mu_y) / sd_y
    K = rbf(x_obs, x_obs) + noise * np.eye(len(x_obs))
    Ks = rbf(x_obs, x_new)
    alpha = np.linalg.solve(K, y)
    mean = Ks.T @ alpha
    var = 1.0 - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
    return mean * sd_y + mu_y, np.sqrt(np.clip(var, 1e-12, None)) * sd_y


def expected_improvement(mean, std, best, xi=0.01):
    """EI = (mu - best - xi) * Phi(z) + sigma * phi(z), with z = (mu - best - xi) / sigma (maximizing)."""
    imp = mean - best - xi
    z = imp / std
    Phi = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    phi = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return imp * Phi + std * phi


def bayes_opt(objective, lo, hi, n_init, n_iter, rng):
    """lo, hi: arrays [d] bounding the (log-scale) box. Candidates: 2,000 random points per step."""
    xs = list(rng.uniform(lo, hi, (n_init, len(lo))))
    ys = [objective(x) for x in xs]
    for _ in range(n_iter):
        cand = rng.uniform(lo, hi, (2000, len(lo)))
        mean, std = gp_posterior(np.array(xs), np.array(ys), cand)
        x_next = cand[np.argmax(expected_improvement(mean, std, max(ys)))]
        xs.append(x_next)
        ys.append(objective(x_next))
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Part 1: grid vs random, 9 trials each, SYNTHETIC surface (best possible 0.900) ===")
    g = grid_points(3)
    print(f"grid learning rates tried: {sorted({float(v) for v in np.round(10 ** g[:, 0], 4)})}")
    g_best = max(score_surface(a, b) for a, b in g)
    print(f"grid best score = {g_best:.4f}")
    gaps = []
    for s in range(2000):
        r = random_points(9, np.random.default_rng(s))
        gaps.append(0.90 - max(score_surface(a, b) for a, b in r))
    print(f"random best score, mean over 2000 seeds = {0.90 - np.mean(gaps):.4f}"
          f"  (beats grid in {np.mean(0.90 - np.array(gaps) > g_best):.0%} of seeds)")
    for n in [9, 30, 60]:
        print(f"P(at least one of {n:2d} random trials lands in the top 5% of the space) = 1 - 0.95^{n} = {1 - 0.95**n:.3f}")

    print("\n=== Part 2: successive halving, 27 configs, eta = 3, SYNTHETIC learning curves ===")
    finals = rng.uniform(0.70, 0.90, 27)
    winner, cost, rounds = successive_halving(finals, rng=np.random.default_rng(1))
    for budget, alive, _ in rounds:
        print(f"  budget {budget:2d} epochs x {len(alive):2d} configs = {budget * len(alive):2d} epochs")
    print(f"total cost = {cost} epochs vs {27 * 9} to train all 27 for 9 epochs")
    print(f"winner final score = {finals[winner]:.3f}, true best = {finals.max():.3f}, "
          f"winner rank = {int((finals > finals[winner]).sum()) + 1} of 27")
    assert cost == 81

    print("\n=== Part 3: Bayesian optimization vs random search on a real SVC (SYNTHETIC data) ===")
    X, y = make_classification(n_samples=400, n_features=10, n_informative=4, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def objective(p):
        log_c, log_gamma = p
        return cross_val_score(SVC(C=10**log_c, gamma=10**log_gamma), X, y, cv=cv).mean()

    lo, hi, budget, seeds = np.array([-2.0, -4.0]), np.array([3.0, 1.0]), 15, 10
    bo_curves, rs_curves = [], []
    for sd in range(seeds):
        _, bo_y = bayes_opt(objective, lo, hi, n_init=4, n_iter=budget - 4, rng=np.random.default_rng(sd))
        rs_y = [objective(p) for p in np.random.default_rng(sd).uniform(lo, hi, (budget, 2))]
        bo_curves.append(np.maximum.accumulate(bo_y))
        rs_curves.append(np.maximum.accumulate(rs_y))
    bo_curve, rs_curve = np.mean(bo_curves, 0), np.mean(rs_curves, 0)
    # Same seed for both: the first 4 points are identical, so any gap after trial 4 is the search strategy.
    print(f"tuning log10 C in [-2, 3] and log10 gamma in [-4, 1], {budget} trials, mean over {seeds} seeds")
    for t in [4, 8, 15]:
        print(f"  after {t:2d} trials: Bayesian opt best = {bo_curve[t - 1]:.4f}   random best = {rs_curve[t - 1]:.4f}")

    print("\n=== Part 4: library search + nested CV (the honest estimate) ===")
    space = {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-4, 1e1)}
    search = RandomizedSearchCV(SVC(), space, n_iter=30, cv=cv, random_state=0)
    search.fit(X, y)
    print(f"RandomizedSearchCV best inner-CV score = {search.best_score_:.4f}  params = "
          f"C={search.best_params_['C']:.3g}, gamma={search.best_params_['gamma']:.3g}")
    outer = StratifiedKFold(5, shuffle=True, random_state=1)
    nested = cross_val_score(RandomizedSearchCV(SVC(), space, n_iter=30, cv=cv, random_state=0), X, y, cv=outer)
    print(f"nested CV score = {nested.mean():.4f} +/- {nested.std():.4f}  (outer folds never seen by the search)")
    grid = GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": [1e-3, 1e-2, 1e-1, 1]}, cv=cv).fit(X, y)
    print(f"GridSearchCV 4x4 best inner-CV score = {grid.best_score_:.4f}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    A, B = np.meshgrid(np.linspace(-3, 0, 200), np.linspace(2, 10, 200))
    ax = axes[0]
    ax.contourf(A, B, score_surface(A, B), levels=12, cmap="Blues")
    r = random_points(9, np.random.default_rng(7))
    ax.scatter(g[:, 0], g[:, 1], s=60, marker="s", color=GRAY, edgecolor="black", label="grid (3 lr values)")
    ax.scatter(r[:, 0], r[:, 1], s=60, color=ORANGE, edgecolor="black", label="random (9 lr values)")
    for v in g[:, 0]:
        ax.plot([v], [1.6], "|", color=GRAY, ms=12)
    for v in r[:, 0]:
        ax.plot([v], [1.6], "|", color=ORANGE, ms=12)
    ax.set(xlabel="log10 learning rate (matters)", ylabel="max depth (barely matters)", xlim=(-3.15, 0.15), ylim=(1.3, 10.5),
           title="9 trials: grid vs random")
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    ax = axes[1]
    for budget_r, alive, scores in rounds:
        ax.scatter([budget_r] * len(alive), scores, color=TEAL, s=25)
    for i in range(27):
        bs = [b for b, al, _ in rounds if i in al]
        ax.plot(bs, [curve(finals[i], b) for b in bs], color=GRAY, lw=0.6, alpha=0.6)
    ax.set_xscale("log", base=3)
    ax.set_xticks([1, 3, 9], ["1", "3", "9"])
    ax.set(xlabel="epochs (budget)", ylabel="validation score", title="Successive halving: 27 → 9 → 3")

    ax = axes[2]
    ax.plot(np.arange(1, budget + 1), bo_curve, "-o", color=BLUE, ms=4, label="Bayesian opt (GP + EI)")
    ax.plot(np.arange(1, budget + 1), rs_curve, "-o", color=ORANGE, ms=4, label="random search")
    ax.axvline(4.5, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="trial", ylabel="best CV accuracy so far", title=f"SVC (C, gamma), mean of {seeds} seeds")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "hyperparameter-tuning")
