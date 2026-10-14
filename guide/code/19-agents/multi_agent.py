# Adapted from AI Engineering from Scratch (MIT), phases/16-multi-agent-and-swarms/01-why-multi-agent
"""Single agent vs orchestrator-workers: a runnable fan-out with MOCKED agents, plus the token and latency arithmetic.

Agents are mocks (no API calls) that report token usage from fixed SYNTHETIC sizes, and "latency" is a short sleep.
Run from guide/:  uv run code/19-agents/multi_agent.py
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

# Synthetic sizes (tokens): single-agent base context, per-step growth, orchestrator base,
# worker base (brief + tools), plan length, summary length per worker.
B, D, B_O, B_W, P, M = 2_000, 1_500, 2_000, 1_500, 300, 400
CALL_SECONDS = 0.05  # mock latency of one model call


def loop_tokens(base, d, steps):
    """Input tokens billed by one agent loop: each call re-reads base + everything appended so far."""
    return steps * base + d * steps * (steps - 1) // 2


def single_agent(k, s):
    t = k * s
    return {"input": loop_tokens(B, D, t), "critical_calls": t, "peak": B + (t - 1) * D}


def single_agent_compacting(k, s):
    """Same single agent, but after each subtask it replaces that subtask's history with an M-token summary."""
    return {"input": sum(s * (B + j * M) + D * s * (s - 1) // 2 for j in range(k)), "critical_calls": k * s}


def orchestrator_workers(k, s):
    plan = B_O
    workers = k * loop_tokens(B_W, D, s)
    synth = B_O + P + k * M
    return {"input": plan + workers + synth, "critical_calls": 1 + s + 1,
            "peak": max(B_W + (s - 1) * D, synth), "parts": (plan, workers, synth)}


# ---------- a runnable fan-out: briefs out, summaries back, nothing else crosses ----------
def worker(brief, steps):
    """A mock worker: its own fresh context (the brief only), a few tool steps, then a short summary."""
    for _ in range(steps):
        time.sleep(CALL_SECONDS)
    return {"to": "orchestrator", "from": brief["id"], "summary": f"findings on {brief['topic']} ({M} tokens)",
            "tokens": loop_tokens(B_W, D, steps)}


def run_fanout(topics, steps):
    briefs = [{"id": f"w{i + 1}", "topic": t, "goal": "3 key facts with sources", "format": "bullets, max 400 tokens"}
              for i, t in enumerate(topics)]
    t0 = time.perf_counter()
    time.sleep(CALL_SECONDS)                                          # orchestrator plans
    with ThreadPoolExecutor(max_workers=len(briefs)) as pool:         # workers run concurrently
        results = list(pool.map(lambda b: worker(b, steps), briefs))
    time.sleep(CALL_SECONDS)                                          # orchestrator synthesizes summaries only
    return results, time.perf_counter() - t0


def figure():
    ks = list(range(1, 11))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ks, [single_agent(k, 3)["input"] / 1e3 for k in ks], "o-", color=ORANGE, label="single agent, 3 steps per subtask")
    a1.plot(ks, [single_agent_compacting(k, 3)["input"] / 1e3 for k in ks], "o:", color=GRAY,
            label="single agent that compacts each subtask")
    a1.plot(ks, [orchestrator_workers(k, 3)["input"] / 1e3 for k in ks], "o-", color=BLUE, label="orchestrator + workers, 3 steps each")
    a1.plot(ks, [orchestrator_workers(k, 6)["input"] / 1e3 for k in ks], "o--", color=TEAL,
            label="workers need 6 steps (re-discover context)")
    a1.set(title="Total input tokens billed", xlabel="independent subtasks k", ylabel="thousand tokens")
    a1.legend(frameon=False, fontsize=8.5)
    a2.plot(ks, [single_agent(k, 3)["critical_calls"] for k in ks], "o-", color=ORANGE, label="single agent")
    a2.plot(ks, [orchestrator_workers(k, 3)["critical_calls"] for k in ks], "o-", color=BLUE, label="orchestrator + workers")
    a2.axhline(0, color=GRAY, lw=0.5)
    a2.set(title="Sequential model calls (latency)", xlabel="independent subtasks k", ylabel="calls on the critical path")
    a2.legend(frameon=False)
    save(fig, "multi-agent-systems")


if __name__ == "__main__":
    k, s = 4, 3
    one, many = single_agent(k, s), orchestrator_workers(k, s)
    print("=== worked example (synthetic sizes) ===")
    print(f"single agent: T = {k}*{s} = {k * s} calls, input = {k * s}*{B} + {D}*{k * s}*{k * s - 1}/2 = {one['input']:,}, "
          f"peak context {one['peak']:,}")
    plan, workers, synth = many["parts"]
    print(f"orchestrator+workers: plan {plan:,} + workers {k}*{loop_tokens(B_W, D, s):,} = {workers:,} + synthesis {synth:,} "
          f"= {many['input']:,}, peak context {many['peak']:,}")
    print(f"critical path: {one['critical_calls']} sequential calls vs {many['critical_calls']}")
    comp = single_agent_compacting(k, s)["input"]
    print(f"single agent that compacts each finished subtask to {M} tokens: {comp:,}")
    dup = orchestrator_workers(k, 6)
    print(f"if each worker needs 6 steps: {dup['input']:,} input tokens (single agent: {one['input']:,})")
    assert (one["input"], many["input"], dup["input"], comp) == (123_000, 41_900, 131_900, 49_200)

    print("\n=== runnable fan-out with mocked workers ===")
    results, secs = run_fanout(["pricing", "competitors", "regulation", "reviews"], steps=s)
    for r in results:
        print(f"{r['from']} -> {r['to']}: {r['summary']}, used {r['tokens']:,} input tokens")
    serial = (1 + k * s + 1) * CALL_SECONDS
    print(f"wall clock {secs:.2f}s vs {serial:.2f}s if the same calls ran one after another")
    figure()
