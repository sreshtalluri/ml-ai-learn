# Adapted from AI Engineering from Scratch (MIT), phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual
"""Group fairness criteria from scratch on SYNTHETIC data with unequal base rates, and why they conflict.

Two groups, base rates 0.3 and 0.6. The score depends only on the true label (same score distribution
given y in both groups). We compare one shared threshold with per-group thresholds chosen for
demographic parity, then check the Chouldechova identity that links FPR, TPR, PPV and the base rate.
Run from guide/:  uv run code/21-safety-security/fairness.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, plt, save  # noqa: E402

rng = np.random.default_rng(21)
N, BASE = 20_000, {"A": 0.3, "B": 0.6}


def make_group(p):
    y = (rng.random(N) < p).astype(int)
    score = rng.normal(1.5 * y, 1.0)          # same P(score | y) in both groups
    return y, score


def rates(y, pred):
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {"sel": (tp + fp) / len(y), "tpr": tp / (tp + fn), "fpr": fp / (fp + tn), "ppv": tp / max(tp + fp, 1)}


def show(name, r):
    print(f"{name:<34}" + "  ".join(f"{g}: sel={v['sel']:.3f} TPR={v['tpr']:.3f} FPR={v['fpr']:.3f} PPV={v['ppv']:.3f}"
                                     for g, v in r.items()))


def threshold_for_rate(score, rate):
    """Threshold that selects the top `rate` fraction of this group's scores."""
    return np.quantile(score, 1 - rate)


def chouldechova_fpr(p, tpr, ppv):
    """FPR implied by base rate p, TPR and PPV:  FPR = p/(1-p) * (1-PPV)/PPV * TPR."""
    return p / (1 - p) * (1 - ppv) / ppv * tpr


def counterfactual_demo():
    """Toy causal model: U (ability, not caused by A) and A both cause the observed test score X = U + 2A."""
    print("\n=== counterfactual fairness in a linear causal model: X = U + 2A ===")
    a, x = 1, 5.0
    u = x - 2 * a                                  # abduction: infer the background variable from what we saw
    x_cf = u + 2 * (1 - a)                         # action + prediction: same U, flip A
    print(f"observed A={a}, X={x}  ->  U = {x} - 2*{a} = {u};  counterfactual X = {u} + 2*{1 - a} = {x_cf}")
    print(f"predictor f(X)=X:  factual {x}, counterfactual {x_cf}  -> not counterfactually fair")
    print(f"predictor g=U:     factual {u}, counterfactual {x_cf - 2 * (1 - a)}  -> counterfactually fair")


def lipschitz_violations(features, decisions, L=1.0):
    """Individual fairness check: count pairs with |f(x)-f(x')| > L * d(x, x'), d = Euclidean distance."""
    bad = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if abs(decisions[i] - decisions[j]) > L * np.linalg.norm(features[i] - features[j]):
                bad += 1
    return bad


def worked_example():
    print("=== worked example: 100 people per group, TPR 0.8 and FPR 0.1 in both ===")
    for g, pos in [("A", 30), ("B", 60)]:
        neg = 100 - pos
        tp, fp = round(0.8 * pos), round(0.1 * neg)
        ppv = tp / (tp + fp)
        print(f"group {g}: positives {pos}, TP={tp}, FP={fp}, selected {tp + fp}/100, PPV={tp}/{tp + fp}={ppv:.3f}, "
              f"identity FPR={chouldechova_fpr(pos / 100, 0.8, ppv):.3f}")


if __name__ == "__main__":
    worked_example()
    counterfactual_demo()
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0]])       # applicants 1 and 2 are near-identical
    print("Lipschitz violations, decisions [0.9, 0.2, 0.1]:", lipschitz_violations(pts, [0.9, 0.2, 0.1]),
          "| decisions [0.9, 0.85, 0.1]:", lipschitz_violations(pts, [0.9, 0.85, 0.1]))
    data = {g: make_group(p) for g, p in BASE.items()}

    print("\n=== synthetic data, 20,000 per group ===")
    shared = {g: rates(y, (s > 0.75).astype(int)) for g, (y, s) in data.items()}
    show("one shared threshold (0.75):", shared)

    target = float(np.mean([shared[g]["sel"] for g in data]))       # equalize selection rates at the average
    dp = {g: rates(y, (s > threshold_for_rate(s, target)).astype(int)) for g, (y, s) in data.items()}
    show(f"per-group thresholds, sel={target:.3f}:", dp)

    gap = {k: abs(dp["A"][k] - dp["B"][k]) for k in ("sel", "tpr", "fpr", "ppv")}
    print("gaps under demographic parity:", {k: round(v, 3) for k, v in gap.items()})
    for g, (y, _) in data.items():
        r = shared[g]
        print(f"identity check group {g}: base rate {y.mean():.3f}, measured FPR {r['fpr']:.3f}, "
              f"implied {chouldechova_fpr(y.mean(), r['tpr'], r['ppv']):.3f}")
    assert abs(shared["A"]["tpr"] - shared["B"]["tpr"]) < 0.02 and gap["sel"] < 0.01 and gap["tpr"] > 0.1

    # Figure: sweep the shared threshold and plot each criterion's gap between the groups.
    ts = np.linspace(-1, 2.5, 36)
    sweep = [{g: rates(y, (s > t).astype(int)) for g, (y, s) in data.items()} for t in ts]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, col, label in [("sel", ORANGE, "selection rate (demographic parity)"),
                            ("tpr", BLUE, "TPR (equal opportunity)"), ("ppv", GRAY, "PPV (predictive parity)")]:
        a1.plot(ts, [r["A"][key] - r["B"][key] for r in sweep], color=col, label=label)
    a1.axhline(0, color="black", lw=0.6)
    a1.set(title="Gap A minus B, one shared threshold", xlabel="threshold on score", ylabel="gap")
    a1.legend(frameon=False, fontsize=8.5)
    keys = ["sel", "tpr", "fpr", "ppv"]
    x = np.arange(len(keys))
    for i, (name, r) in enumerate([("shared threshold", shared), ("demographic-parity thresholds", dp)]):
        a2.bar(x + (i - 0.5) * 0.38, [r["A"][k] - r["B"][k] for k in keys], width=0.38,
               color=[BLUE, ORANGE][i], label=name)
    a2.axhline(0, color="black", lw=0.6)
    a2.set_xticks(x, ["selection", "TPR", "FPR", "PPV"])
    a2.set(title="Fixing one gap opens others", ylabel="gap A minus B")
    a2.legend(frameon=False, fontsize=8.5)
    save(fig, "fairness")
