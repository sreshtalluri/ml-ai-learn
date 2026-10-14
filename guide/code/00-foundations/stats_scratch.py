# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/06-probability-and-distributions, 15-statistics-for-ml
"""Descriptive statistics, Bayes' rule, the central limit theorem, a bootstrap interval, and a
permutation test, from scratch. All data is SYNTHETIC and seeded.

Run from guide/:  uv run code/00-foundations/stats_scratch.py
"""
import random
import statistics


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs, sample=True):
    m = mean(xs)
    ss = sum((x - m) ** 2 for x in xs)
    return ss / (len(xs) - 1 if sample else len(xs))   # n - 1: Bessel's correction


def pearson(x, y):
    mx, my = mean(x), mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5


def bayes(prior, sensitivity, false_positive_rate):
    evidence = sensitivity * prior + false_positive_rate * (1 - prior)   # P(positive)
    return sensitivity * prior / evidence


def bootstrap_ci(xs, stat=mean, n_boot=2000, alpha=0.05, rng=random):
    boots = sorted(stat(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return boots[int(n_boot * alpha / 2)], boots[int(n_boot * (1 - alpha / 2)) - 1]


def permutation_test(a, b, n_perm=5000, rng=random):
    """Two-sided p-value for 'mean(a) - mean(b) is zero': shuffle the labels and count."""
    observed = abs(mean(a) - mean(b))
    pooled = a + b
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(mean(pooled[:len(a)]) - mean(pooled[len(a):])) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


if __name__ == "__main__":
    rng = random.Random(0)
    xs = [2, 4, 4, 4, 5, 5, 7, 9]
    print(f"mean {mean(xs)}, population var {variance(xs, sample=False)}, sample var {variance(xs):.4f}")

    print(f"P(disease | positive) = {bayes(0.01, 0.95, 0.05):.4f}")

    # Central limit theorem: means of 30 die rolls are close to normal, with std = σ/√30
    means = [mean([rng.randint(1, 6) for _ in range(30)]) for _ in range(5000)]
    sigma = variance(list(range(1, 7)), sample=False) ** 0.5
    print(f"CLT: std of 5000 sample means {statistics.pstdev(means):.4f}, predicted σ/√30 = {sigma / 30**0.5:.4f}")

    # Bootstrap 95% interval for accuracy: 870 of 1000 correct
    correct = [1] * 870 + [0] * 130
    lo, hi = bootstrap_ci(correct, rng=rng)
    print(f"accuracy 0.870, bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")

    # Permutation test: model A vs model B on per-example scores (SYNTHETIC)
    a = [rng.gauss(0.80, 0.10) for _ in range(50)]
    b = [rng.gauss(0.75, 0.10) for _ in range(50)]
    print(f"mean A {mean(a):.3f}, mean B {mean(b):.3f}, permutation p = {permutation_test(a, b, rng=rng):.4f}")

    temp = [rng.gauss(25, 6) for _ in range(300)]
    ice = [20 + 3 * t + rng.gauss(0, 10) for t in temp]
    swim = [2 + 0.15 * t + rng.gauss(0, 1) for t in temp]
    print(f"r(ice cream, incidents) = {pearson(ice, swim):.3f}  (confounder: temperature)")

    assert abs(variance(xs) - statistics.variance(xs)) < 1e-12
    assert abs(pearson(temp, ice) - statistics.correlation(temp, ice)) < 1e-9
    assert abs(statistics.pstdev(means) - sigma / 30**0.5) < 0.02
    print("all checks match the statistics module")
