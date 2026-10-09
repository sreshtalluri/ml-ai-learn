"""Agent reliability arithmetic: how per-step errors compound, what verification + retry buys,
pass@k vs pass^k, and the cost/latency of a multi-agent fan-out versus a single agent.

All probabilities, token counts, timings, and prices are SYNTHETIC illustrations. Deterministic.
Run from guide/:  uv run code/19-agents/agent_reliability.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def task_success(p_step, n_steps):
    """Independent steps that must all succeed."""
    return p_step ** n_steps


def with_verify_retry(p_step, recall, retries=1):
    """A step fails only if it errs AND (the check misses it OR every retry also errs).
    recall = fraction of bad steps the verifier catches."""
    q = 1 - p_step
    return 1 - q * ((1 - recall) + recall * q ** retries)


def pass_at_k(p, k):    # at least one of k independent trials succeeds
    return 1 - (1 - p) ** k


def pass_hat_k(p, k):   # all k trials succeed (consistency)
    return p ** k


def simulate(p_step, n_steps, trials=20_000, seed=0):
    rng = np.random.default_rng(seed)
    return float((rng.random((trials, n_steps)) < p_step).all(axis=1).mean())


if __name__ == "__main__":
    p, n = 0.95, 10
    s = task_success(p, n)
    print(f"Per-step success {p}, {n} steps: {p}^{n} = {s:.4f}  (Monte Carlo: {simulate(p, n):.4f})")
    pv = with_verify_retry(p, recall=0.8)
    print(f"Verifier recall 0.8, one retry: per-step = 1 - 0.05*(0.2 + 0.8*0.05) = {pv:.4f}; "
          f"task = {pv:.4f}^{n} = {task_success(pv, n):.4f}")
    print(f"Fewer steps: {p}^5 = {task_success(p, 5):.4f}")
    t = 0.7
    print(f"Per-trial task success {t}: pass@3 = 1 - 0.3^3 = {pass_at_k(t, 3):.3f}; pass^3 = 0.7^3 = {pass_hat_k(t, 3):.3f}")

    # Multi-agent vs single agent (SYNTHETIC)
    sub_s, sub_tok = 40, 25_000            # one research subtask: seconds, tokens (input+output)
    single_lat, single_tok = 3 * sub_s, 3 * sub_tok + 2 * 15_000  # later subtasks re-read earlier findings
    orch_s, orch_tok, synth_s, synth_tok = 8, 6_000, 12, 20_000
    multi_lat = orch_s + sub_s + synth_s     # workers run in parallel
    worker_tok = 35_000                      # brief + own exploration, some searches duplicated across workers
    multi_tok = orch_tok + 3 * worker_tok + synth_tok
    print(f"Single agent: latency {single_lat}s, tokens {single_tok:,}")
    print(f"Orchestrator + 3 parallel workers: latency {orch_s}+{sub_s}+{synth_s} = {multi_lat}s, "
          f"tokens {orch_tok:,} + 3*{worker_tok:,} + {synth_tok:,} = {multi_tok:,} ({multi_tok / single_tok:.2f}x)")

    # ---------- figure ----------
    steps = np.arange(1, 31)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ps, c in [(0.99, TEAL), (0.95, BLUE), (0.90, ORANGE)]:
        a.plot(steps, task_success(ps, steps), color=c, label=f"per-step {ps}")
    a.plot(steps, task_success(pv, steps), color=PURPLE, ls="--", label=f"0.95 + verify/retry ({pv:.3f})")
    a.scatter([n], [s], color=BLUE, zorder=3)
    a.annotate(f"0.95^10 = {s:.2f}", (n, s), xytext=(13, 0.72), arrowprops={"arrowstyle": "->"})
    a.set(xlabel="steps that must all succeed", ylabel="task success", ylim=(0, 1.02),
          title="Errors compound over steps")
    a.legend(fontsize=8, loc="lower left")
    labels = ["single agent", "orchestrator\n+ 3 workers"]
    x = np.arange(2)
    b.bar(x - 0.2, [single_lat, multi_lat], 0.4, color=BLUE, label="latency (s)")
    b2 = b.twinx()
    b2.bar(x + 0.2, [single_tok / 1000, multi_tok / 1000], 0.4, color=ORANGE, label="tokens (k)")
    b2.grid(False)
    b.set_xticks(x, labels)
    b.set(ylabel="latency (s)", title="Multi-agent: faster wall clock, more tokens")
    b2.set_ylabel("tokens (thousands)")
    b.legend(loc="upper left", fontsize=8)
    b2.legend(loc="upper right", fontsize=8)
    b2.set_ylim(0, 160)
    b.set_ylim(0, 140)
    fig.subplots_adjust(wspace=0.4)
    save(fig, "reliable-agents")
