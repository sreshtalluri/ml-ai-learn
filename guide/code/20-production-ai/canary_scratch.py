# Adapted from AI Engineering from Scratch (MIT), phases/11-llm-engineering/13-production-app,
# phases/17-infrastructure-and-production/20-shadow-canary-progressive
"""A production request path, shadow comparison, and a gated canary rollout, from scratch.

Everything is simulated: models are functions with SYNTHETIC latency, cost, and error rates.
Deterministic (fixed seeds). Stdlib only.
Run from guide/:  uv run code/20-production-ai/canary_scratch.py
"""
import random
import statistics


def make_model(name, latency_ms, cost, error_rate, length):
    """A fake model: returns (text, latency_ms, cost_usd) or raises on a simulated failure."""
    def call(prompt, rng):
        if rng.random() < error_rate:
            raise RuntimeError(f"{name}: 503")
        lat = latency_ms * rng.uniform(0.7, 1.6)
        return f"{name} answer to {prompt!r}"[: length], lat, cost
    call.name = name
    return call


def handle(prompt, primary, fallback, rng, timeout_ms=2000):
    """One request: try the primary under a deadline, fall back once, never hang."""
    for model in (primary, fallback):
        try:
            text, lat, cost = model(prompt, rng)
        except RuntimeError:
            continue                      # dependency failed: try the next model
        if lat > timeout_ms:
            continue                      # too slow: treat as a timeout
        return {"model": model.name, "text": text, "latency_ms": lat, "cost": cost}
    return {"model": "none", "text": "Sorry, try again shortly.", "latency_ms": timeout_ms, "cost": 0.0}


def shadow(prompts, prod, cand, seed=0):
    """Send the same prompts to both; users only ever see prod. Log the deltas."""
    rng = random.Random(seed)
    rows = []
    for p in prompts:
        a = handle(p, prod, prod, rng)
        b = handle(p, cand, cand, rng)        # candidate output is logged, never returned
        rows.append((a["latency_ms"], b["latency_ms"], a["cost"], b["cost"], b["model"] == "none"))
    return {
        "latency_ratio": statistics.median(r[1] for r in rows) / statistics.median(r[0] for r in rows),
        "cost_ratio": sum(r[3] for r in rows) / sum(r[2] for r in rows),
        "cand_fail_rate": sum(r[4] for r in rows) / len(rows),
    }


STAGES = [0.01, 0.10, 0.25, 0.50, 1.00]
GATES = {"p95_ms": 1.5, "cost": 1.2, "error_rate": 2.0}   # canary may be at most this multiple of baseline


def p95(xs):
    xs = sorted(xs)
    return xs[int(0.95 * (len(xs) - 1))]


def measure(model, n, rng):
    lats, costs, errs = [], [], 0
    for i in range(n):
        try:
            _, lat, cost = model(f"q{i}", rng)
            lats.append(lat)
            costs.append(cost)
        except RuntimeError:
            errs += 1
    return {"p95_ms": p95(lats), "cost": statistics.mean(costs), "error_rate": max(errs / n, 1e-3)}


def canary(baseline, cand, requests_per_stage=4000, seed=1):
    """Shift traffic stage by stage; compare canary to baseline at each stage; roll back on any breach."""
    rng = random.Random(seed)
    for share in STAGES:
        n_cand = max(200, int(share * requests_per_stage))      # floor so early stages have signal
        base, can = measure(baseline, requests_per_stage, rng), measure(cand, n_cand, rng)
        breaches = [k for k, mult in GATES.items() if can[k] > base[k] * mult]
        print(f"  {share:>4.0%}  p95 {can['p95_ms']:6.0f} vs {base['p95_ms']:6.0f} ms  "
              f"err {can['error_rate']:.3f} vs {base['error_rate']:.3f}  "
              f"{'ROLLBACK: ' + ','.join(breaches) if breaches else 'pass'}")
        if breaches:
            return "rolled back"
    return "promoted"


if __name__ == "__main__":
    prod = make_model("prod-v1", latency_ms=800, cost=0.010, error_rate=0.01, length=60)
    small = make_model("small-v1", latency_ms=300, cost=0.002, error_rate=0.01, length=60)
    flaky = make_model("flaky-v1", latency_ms=900, cost=0.010, error_rate=0.40, length=60)

    print("Request path (primary flaky, fallback small):")
    rng = random.Random(0)
    served = [handle(f"q{i}", flaky, small, rng)["model"] for i in range(1000)]
    counts = {m: served.count(m) for m in sorted(set(served))}
    print("  served by:", counts)
    assert counts.get("flaky-v1", 0) + counts.get("small-v1", 0) + counts.get("none", 0) == 1000
    assert counts.get("none", 0) < 20                 # fallback keeps almost every request answered

    print("Shadow (candidate = cheaper, faster model):")
    s = shadow([f"q{i}" for i in range(500)], prod, small)
    print(f"  latency ratio {s['latency_ratio']:.2f}  cost ratio {s['cost_ratio']:.2f}  "
          f"candidate failures {s['cand_fail_rate']:.1%}")
    assert s["cost_ratio"] < 0.3

    good = make_model("cand-good", latency_ms=820, cost=0.011, error_rate=0.01, length=60)
    slow = make_model("cand-slow", latency_ms=1400, cost=0.011, error_rate=0.01, length=60)
    pricey = make_model("cand-pricey", latency_ms=800, cost=0.016, error_rate=0.01, length=60)
    results = {}
    for cand in (good, slow, pricey):
        print(f"Canary {cand.name}:")
        results[cand.name] = canary(prod, cand)
        print("  ->", results[cand.name])
    assert results == {"cand-good": "promoted", "cand-slow": "rolled back", "cand-pricey": "rolled back"}
    print("all checks passed")
