# Adapted from AI Engineering from Scratch (MIT), phases/17-infrastructure-and-production/21-ab-testing-llm-features
"""Peeking-safe A/B testing from scratch: an always-valid mSPRT, Benjamini-Hochberg, and an SRM check.

All traffic is SYNTHETIC (seeded NumPy). The mSPRT follows Johari et al., "Always Valid Inference" (2017),
with a normal mixing prior on the effect.
Run from guide/:  uv run code/22-ml-system-design/sequential_test_scratch.py
"""
import math
from statistics import NormalDist

import numpy as np

N = NormalDist()


# --- 1. Naive peeking: a z-test re-run after every day ----------------------------------------------
def z_stats(ca, cb, n):
    """Two-proportion z per look. ca, cb: cumulative conversions [looks]; n: cumulative users per arm."""
    pool = (ca + cb) / (2 * n)
    se = np.sqrt(pool * (1 - pool) * 2 / n)
    return np.where(se > 0, (cb - ca) / n / np.where(se > 0, se, 1), 0.0)


# --- 2. mSPRT: a likelihood ratio averaged over plausible effects; stop when it reaches 1/alpha ----
def msprt_lambda(ca, cb, n, tau2):
    """Mixture likelihood ratio for H0: no difference. V is the variance of the observed difference."""
    pool = (ca + cb) / (2 * n)
    V = np.maximum(2 * pool * (1 - pool) / n, 1e-12)
    d = (cb - ca) / n
    return np.sqrt(V / (V + tau2)) * np.exp(tau2 * d**2 / (2 * V * (V + tau2)))


# --- 3. Simulate many experiments, look once per day, record whether each rule ever fires --------
def simulate(p_a, p_b, users_per_day, days, sims, alpha=0.05, tau2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    ca = rng.binomial(users_per_day, p_a, size=(sims, days)).cumsum(1)
    cb = rng.binomial(users_per_day, p_b, size=(sims, days)).cumsum(1)
    n = users_per_day * np.arange(1, days + 1)
    zcrit = N.inv_cdf(1 - alpha / 2)
    naive = (np.abs(z_stats(ca, cb, n)) > zcrit).any(1).mean()
    final = (np.abs(z_stats(ca, cb, n))[:, -1] > zcrit).mean()
    hit = msprt_lambda(ca, cb, n, tau2) >= 1 / alpha
    stop_day = np.where(hit.any(1), hit.argmax(1) + 1, days + 1)
    return final, naive, hit.any(1).mean(), stop_day


# --- 4. Benjamini-Hochberg: control the false-discovery rate across many metrics or tests --------
def benjamini_hochberg(pvals, q=0.05):
    order = sorted(range(len(pvals)), key=pvals.__getitem__)
    k = max((r for r, i in enumerate(order, 1) if pvals[i] <= q * r / len(pvals)), default=0)
    return sorted(order[:k])                          # indices declared significant


# --- 5. Sample ratio mismatch: chi-square test of the observed split against the planned one ----
def srm_pvalue(n_a, n_b, share_a=0.5):
    total = n_a + n_b
    chi2 = (n_a - total * share_a) ** 2 / (total * share_a) + (n_b - total * (1 - share_a)) ** 2 / (total * (1 - share_a))
    return 2 * (1 - N.cdf(math.sqrt(chi2)))           # 1 degree of freedom: chi2 = z^2


if __name__ == "__main__":
    final, naive, msprt, _ = simulate(0.10, 0.10, 1500, 14, 4000, seed=7)
    print("A/A (no true effect), 4,000 synthetic experiments, 14 daily looks, 1,500 users/arm/day")
    print(f"  fixed z-test, one look at day 14 : false-positive rate {final:.3f}")
    print(f"  z-test, stop at first p < 0.05   : false-positive rate {naive:.3f}")
    print(f"  mSPRT, stop when Lambda >= 20    : false-positive rate {msprt:.3f}")

    power, _, msprt_power, stop = simulate(0.10, 0.11, 1500, 14, 4000, seed=8)
    print("A/B with a true 10% relative lift (10% -> 11%)")
    print(f"  fixed z-test power at day 14 {power:.3f}; mSPRT power {msprt_power:.3f}, "
          f"median stop day {np.median(stop):.0f} (15 = never stopped)")

    pvals = [0.001, 0.008, 0.012, 0.041, 0.049, 0.20, 0.35, 0.62, 0.74, 0.91]
    print(f"10 metrics, raw p < 0.05: {[i for i, p in enumerate(pvals) if p < 0.05]}")
    print(f"  Bonferroni (p < 0.005): {[i for i, p in enumerate(pvals) if p < 0.05 / len(pvals)]}")
    print(f"  Benjamini-Hochberg q=0.05: {benjamini_hochberg(pvals)}")
    print(f"SRM 50,600 vs 49,400 planned 50/50: p = {srm_pvalue(50_600, 49_400):.5f}")

    assert naive > 0.1 and msprt <= 0.05
    assert benjamini_hochberg(pvals) == [0, 1, 2]
