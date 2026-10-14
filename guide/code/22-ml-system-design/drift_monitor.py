# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/20-model-monitoring-and-drift
"""Drift monitoring from scratch: PSI, the two-sample KS statistic, chi-square with Cramer's V, sliding windows,
delayed labels, and an alert policy with persistence. Runs on a SYNTHETIC payment stream with two injected changes:
covariate drift at event 3000 (bigger baskets, more app traffic) and concept drift at event 6500 (fraud boundary moves).
Run from guide/:  uv run code/22-ml-system-design/drift_monitor.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

CHANNELS = ("app", "store", "web")
# (start event, amount mean, amount sd, channel mix, fraud boundary on amount)
SEGMENTS = [(0, 100, 20, (0.35, 0.15, 0.50), 120), (3000, 118, 24, (0.60, 0.10, 0.30), 120),
            (5000, 100, 20, (0.35, 0.15, 0.50), 120), (6500, 100, 20, (0.35, 0.15, 0.50), 105)]


def make_stream(n=9000, seed=7):
    rng = np.random.default_rng(seed)
    amount, channel, label = np.empty(n), np.empty(n, dtype=object), np.empty(n, dtype=int)
    for i, (start, mu, sd, mix, boundary) in enumerate(SEGMENTS):
        end = SEGMENTS[i + 1][0] if i + 1 < len(SEGMENTS) else n
        k = end - start
        amount[start:end] = rng.normal(mu, sd, k)
        channel[start:end] = rng.choice(CHANNELS, size=k, p=mix)
        label[start:end] = (amount[start:end] + rng.normal(0, 6, k) > boundary).astype(int)
    return amount, channel, label


# ---------- the three statistics ----------
def psi(reference, current, bins=10, eps=1e-4):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])      # reference deciles
    e = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=bins) / len(reference)
    a = np.bincount(np.searchsorted(edges, current, side="right"), minlength=bins) / len(current)
    e, a = np.maximum(e, eps), np.maximum(a, eps)                            # avoid log(0)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(x, y):
    """Largest vertical gap between the two empirical CDFs, evaluated at every observed value."""
    grid = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(np.sort(x), grid, side="right") / len(x)
    fy = np.searchsorted(np.sort(y), grid, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def chi_square(ref_counts, cur_counts):
    table = np.array([ref_counts, cur_counts], dtype=float)                 # [2, k categories]
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    dof = table.shape[1] - 1
    return chi2, float(stats.chi2.sf(chi2, dof)), float(np.sqrt(chi2 / table.sum()))  # Cramer's V for 2 rows


def counts(values):
    return [int((values == c).sum()) for c in CHANNELS]


# ---------- alert policy: effect size + significance + persistence ----------
class Policy:
    def __init__(self, n_tests, family_alpha=0.01, psi_drift=0.25, ks_drift=0.10, min_v=0.10, patience=2):
        self.alpha, self.psi_drift, self.ks_drift, self.min_v, self.patience = family_alpha / n_tests, psi_drift, ks_drift, min_v, patience
        self.streak = {}

    def update(self, name, drifting):
        prev = self.streak.get(name, 0)
        self.streak[name] = prev + 1 if drifting else 0
        if self.streak[name] == self.patience:
            return "ALERT"
        if prev >= self.patience and not drifting:
            return "resolved"
        return None


def monitor(amount, channel, label, ref_size=2000, window=500, step=250, delay=500, tol=0.05):
    ref_amt, ref_ch = amount[:ref_size], counts(channel[:ref_size])
    # The "model": one threshold on amount, fitted on the reference period.
    cands = np.sort(ref_amt)[::5]
    thr = cands[np.argmax([((ref_amt > t) == label[:ref_size]).mean() for t in cands])]
    pred = (amount > thr).astype(int)
    ref_acc = float((pred[:ref_size] == label[:ref_size]).mean())
    policy, rows = Policy(n_tests=2), []
    for end in range(ref_size + window, len(amount) + 1, step):
        cur = slice(end - window, end)
        p, d = psi(ref_amt, amount[cur]), ks_statistic(ref_amt, amount[cur])
        ks_p = float(stats.kstwo.sf(d, round(ref_size * window / (ref_size + window))))
        chi2, chi_p, v = chi_square(ref_ch, counts(channel[cur]))
        known = slice(end - delay - window, end - delay)                     # labels only exist `delay` events back
        acc = float((pred[known] == label[known]).mean())
        flags = {"amount": p >= policy.psi_drift or (d >= policy.ks_drift and ks_p < policy.alpha),
                 "channel": chi_p < policy.alpha and v >= policy.min_v,
                 "accuracy": acc < ref_acc - tol}
        notes = [f"{o} {k}" for k, f in flags.items() if (o := policy.update(k, f))]
        rows.append(dict(end=end, psi=p, ks=d, ks_p=ks_p, chi_p=chi_p, v=v, acc=acc, notes=notes))
    return rows, float(thr), ref_acc


def worked_psi():
    print("=== worked PSI: 3 of 10 decile bins move (synthetic shares) ===")
    ref = [0.10] * 10
    cur = [0.05, 0.05, 0.20] + [0.10] * 7
    terms = [(a - e) * np.log(a / e) for a, e in zip(cur, ref)]
    for i in range(3):
        print(f"bin {i + 1}: ({cur[i]:.2f} - {ref[i]:.2f}) * ln({cur[i]:.2f}/{ref[i]:.2f}) = {terms[i]:.4f}")
    print(f"PSI = {sum(terms):.4f}")
    return sum(terms)


if __name__ == "__main__":
    assert abs(worked_psi() - 0.1386) < 1e-4
    amount, channel, label = make_stream()
    rows, thr, ref_acc = monitor(amount, channel, label)
    print(f"\nmodel threshold {thr:.1f}, reference accuracy {ref_acc:.3f}")
    print(f"{'end':>5} {'PSI':>6} {'KS D':>6} {'chi2 p':>8} {'V':>5} {'acc':>6}  events")
    for r in rows:
        if r["notes"] or r["end"] in (2500, 3250, 7000):
            print(f"{r['end']:>5} {r['psi']:6.3f} {r['ks']:6.3f} {r['chi_p']:8.1e} {r['v']:5.3f} {r['acc']:6.3f}  {', '.join(r['notes'])}")

    print("\n=== library check (scipy) on the window ending at 4000 ===")
    cur = slice(3500, 4000)
    print(f"scratch KS D = {ks_statistic(amount[:2000], amount[cur]):.4f};  scipy ks_2samp D = "
          f"{stats.ks_2samp(amount[:2000], amount[cur]).statistic:.4f}")
    t = [counts(channel[:2000]), counts(channel[cur])]
    print(f"scratch chi2 = {chi_square(*t)[0]:.2f};  scipy chi2_contingency = {stats.chi2_contingency(t).statistic:.2f}")

    ends = [r["end"] for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    a1.plot(ends, [r["psi"] for r in rows], color=BLUE, label="PSI (amount)")
    a1.plot(ends, [r["ks"] for r in rows], color=TEAL, label="KS D (amount)")
    a1.plot(ends, [r["v"] for r in rows], color=PURPLE, label="Cramer's V (channel)")
    a1.axhline(0.25, color=GRAY, ls="--", lw=1)
    a1.text(2100, 0.27, "PSI 0.25", color=GRAY, fontsize=8)
    a1.set(title="Input monitors (no labels needed)", ylabel="statistic")
    a1.legend(frameon=False, fontsize=8.5, loc="center right")
    a2.plot(ends, [r["acc"] for r in rows], color=ORANGE, label="accuracy on labelled events (500-event delay)")
    a2.axhline(ref_acc - 0.05, color=GRAY, ls="--", lw=1)
    a2.text(2100, ref_acc - 0.045, "alert line", color=GRAY, fontsize=8)
    a2.set(title="Outcome monitor (needs labels)", xlabel="event index (window end)", ylabel="accuracy")
    a2.legend(frameon=False, fontsize=8.5, loc="lower left")
    for ax in (a1, a2):
        ax.axvspan(3000, 5000, color=BLUE, alpha=0.07)
        ax.axvspan(6500, 9000, color=ORANGE, alpha=0.07)
    a1.text(3050, a1.get_ylim()[1] * 0.9, "covariate drift", color=BLUE, fontsize=8.5)
    a1.text(6550, a1.get_ylim()[1] * 0.9, "concept drift", color=ORANGE, fontsize=8.5)
    save(fig, "model-monitoring-and-drift")
