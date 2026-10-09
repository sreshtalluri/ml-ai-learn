"""A/B test arithmetic: sample size, a two-proportion z-test, SRM check, CUPED, and a seeded simulation of how
peeking inflates false positives. All data is SYNTHETIC.

Run from guide/:  uv run code/22-ml-system-design/ab_test.py
"""
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

N = NormalDist()


def sample_size(p1, rel_mde, alpha=0.05, power=0.8):
    """Per-arm n for a two-sided two-proportion z-test (unpooled variance)."""
    p2 = p1 * (1 + rel_mde)
    z = N.inv_cdf(1 - alpha / 2) + N.inv_cdf(power)
    return math.ceil(z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)


def power_at(p1, rel_lift, n, alpha=0.05):
    p2 = p1 * (1 + rel_lift)
    se = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return N.cdf(abs(p2 - p1) / se - N.inv_cdf(1 - alpha / 2))


def z_test(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (x2 / n2 - x1 / n1) / se
    return p, se, z, 2 * (1 - N.cdf(abs(z)))


def aa_false_positive_rate(p, per_arm_per_day, days, sims, alpha=0.05, seed=0):
    """Run `sims` A/A tests (no true effect). Return (FPR with one look at the end, FPR when stopping at the first
    daily look with p < alpha, FPR curve vs number of looks)."""
    rng = np.random.default_rng(seed)
    c = rng.binomial(per_arm_per_day, p, size=(sims, days)).cumsum(1)
    t = rng.binomial(per_arm_per_day, p, size=(sims, days)).cumsum(1)
    n = per_arm_per_day * np.arange(1, days + 1)
    pool = (c + t) / (2 * n)
    z = (t - c) / n / np.sqrt(pool * (1 - pool) * 2 / n)
    sig = np.abs(z) > N.inv_cdf(1 - alpha / 2)
    ever = np.logical_or.accumulate(sig, axis=1).mean(0)  # FPR if you peek on days 1..d
    return sig[:, -1].mean(), ever[-1], ever


if __name__ == "__main__":
    za, zb = N.inv_cdf(0.975), N.inv_cdf(0.8)
    p1, p2 = 0.10, 0.11
    var = p1 * (1 - p1) + p2 * (1 - p2)
    raw = (za + zb) ** 2 * var / (p2 - p1) ** 2
    print("Sample size: baseline 10%, relative MDE 10% (10% -> 11%), alpha 0.05 two-sided, power 0.8")
    print(f"  z_(1-a/2) = {za:.4f}, z_(power) = {zb:.4f}, sum = {za + zb:.4f}, squared = {(za + zb) ** 2:.4f}")
    print(f"  variance term = 0.10*0.90 + 0.11*0.89 = {var:.4f}; delta^2 = {(p2 - p1) ** 2:.4f}")
    print(f"  n = {(za + zb) ** 2:.4f} * {var:.4f} / 0.0001 = {raw:.1f} -> {sample_size(0.10, 0.10)} per arm")
    n = sample_size(0.10, 0.10)
    print(f"  at 3,000 eligible users/day (1,500 per arm): {n} / 1500 = {n / 1500:.2f} -> {math.ceil(n / 1500)} days")
    print(f"  power at a 5% relative lift with that n: {power_at(0.10, 0.05, n):.3f}")

    print("\nTwo-proportion z-test on synthetic results: 1,500/15,000 vs 1,650/15,000")
    p, se, z, pv = z_test(1500, 15000, 1650, 15000)
    print(f"  pooled p = {p:.4f}, SE = {se:.6f}, z = {z:.3f}, two-sided p-value = {pv:.4f}")
    print(f"  95% CI for the difference (unpooled): 0.01 +/- 1.96 * "
          f"{math.sqrt(0.1 * 0.9 / 15000 + 0.11 * 0.89 / 15000):.5f}")

    obs = (50_600, 49_400)
    chi2 = sum((o - 50_000) ** 2 / 50_000 for o in obs)
    print(f"\nSRM check 50,600 vs 49,400 (expected 50/50): chi^2 = {chi2:.1f}, "
          f"p = {2 * (1 - N.cdf(math.sqrt(chi2))):.5f}")
    print(f"CUPED with rho = 0.5: variance x (1 - 0.25) = 0.75 -> n {n} -> {math.ceil(0.75 * n)}")

    fpr_final, fpr_peek, curve = aa_false_positive_rate(0.10, 1500, 14, 4000, seed=7)
    print(f"\nA/A simulation (4,000 tests, 14 days, 1,500 users/arm/day, seed 7):")
    print(f"  one look at day 14: false-positive rate {fpr_final:.3f}")
    print(f"  stop at first daily p < 0.05: false-positive rate {fpr_peek:.3f}")

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    lifts = np.linspace(0, 0.2, 101)
    a1.plot(lifts * 100, [power_at(0.10, l, n) for l in lifts], color=BLUE, label=f"n = {n:,} per arm")
    a1.plot(lifts * 100, [power_at(0.10, l, n // 2) for l in lifts], color=TEAL, ls="--", label=f"n = {n // 2:,} per arm")
    a1.axhline(0.8, color=GRAY, lw=1)
    a1.axvline(10, color=ORANGE, lw=1.5, ls=":")
    a1.text(10.3, 0.1, "MDE 10%", color=ORANGE)
    a1.set(xlabel="true relative lift (%)", ylabel="power", ylim=(0, 1.02),
           title="Power curve (baseline 10%, alpha 0.05)")
    a1.legend(loc="upper left", fontsize=8)
    days = np.arange(1, 15)
    a2.plot(days, curve, marker="o", color=ORANGE, label="peek daily, stop at first p < 0.05")
    a2.axhline(0.05, color=BLUE, ls="--", label="nominal alpha = 0.05")
    a2.set(xlabel="number of daily looks", ylabel="false-positive rate (A/A tests)", ylim=(0, max(curve) * 1.2),
           title="Peeking inflates false positives")
    a2.legend(loc="upper left", fontsize=8)
    save(fig, "experimentation-and-ab-testing")
