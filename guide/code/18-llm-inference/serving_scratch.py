# Adapted from AI Engineering from Scratch (MIT), phases/17-infrastructure-and-production/04-vllm-serving-internals, phases/17-infrastructure-and-production/06-sglang-radixattention, phases/17-infrastructure-and-production/08-inference-metrics-goodput
"""Serving internals from scratch (stdlib only): a paged KV block allocator, a block-hash prefix cache,
and TTFT / TPOT / goodput from request traces. All requests and latencies are SYNTHETIC.
Run from guide/:  uv run code/18-llm-inference/serving_scratch.py
"""
import random
import statistics

BLOCK = 16  # tokens per KV block


# ---------- 1. Paged KV allocator: hand out fixed-size blocks, not max-length slabs ----------
class BlockPool:
    def __init__(self, n_blocks):
        self.free = list(range(n_blocks))
        self.tables = {}  # request id -> list of physical block ids (its "page table")

    def grow(self, rid, n_tokens):
        """Make sure request rid owns enough blocks for n_tokens. False means: preempt someone."""
        table = self.tables.setdefault(rid, [])
        need = -(-n_tokens // BLOCK) - len(table)  # ceil division
        if need > len(self.free):
            return False
        table += [self.free.pop() for _ in range(need)]
        return True

    def release(self, rid):
        self.free += self.tables.pop(rid, [])


# ---------- 2. Prefix cache: hash each full block together with everything before it ----------
def block_hashes(tokens):
    h, out = 0, []
    for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
        h = hash((h, tuple(tokens[i : i + BLOCK])))  # chain: same hash only if the whole prefix matches
        out.append(h)
    return out


def prefill_with_cache(cache, tokens):
    """Return how many prompt tokens were served from cache, then store this prompt's blocks."""
    hit = 0
    for h in block_hashes(tokens):
        if h not in cache:
            break  # first miss ends the shared prefix
        hit += BLOCK
    cache.update(block_hashes(tokens))
    return hit


# ---------- 3. Latency metrics and goodput ----------
def metrics(trace):
    """trace = (send time, list of token arrival times). TPOT excludes the first token."""
    t0, toks = trace
    ttft = toks[0] - t0
    tpot = (toks[-1] - toks[0]) / (len(toks) - 1)
    return ttft, tpot, toks[-1] - t0


def goodput(traces, ttft_slo, tpot_slo):
    ok = [m[0] <= ttft_slo and m[1] <= tpot_slo for m in map(metrics, traces)]
    return sum(ok) / len(ok)  # a request counts only if it met EVERY SLO


if __name__ == "__main__":
    rng = random.Random(7)
    lens = [rng.randint(100, 1100) for _ in range(200)]  # synthetic context lengths, mean ~600
    pool, admitted = BlockPool(4000), 0
    for rid, n in enumerate(lens):
        admitted += pool.grow(rid, n)
    used = 4000 - len(pool.free)
    print(f"[paged] 4000 blocks: admitted {admitted}/200 requests, {used} blocks used, "
          f"waste {used * BLOCK - sum(lens[:admitted])} tokens")
    print(f"[paged] contiguous 4096-token slabs fit {4000 * BLOCK // 4096} requests")
    pool.release(0)
    assert len(pool.free) == 4000 - used + -(-lens[0] // BLOCK)

    system = [rng.randrange(32000) for _ in range(2000)]  # shared synthetic system prompt
    cache, total, hit = set(), 0, 0
    for _ in range(50):
        prompt = system + [rng.randrange(32000) for _ in range(200)]
        hit += prefill_with_cache(cache, prompt)
        total += len(prompt)
    print(f"[prefix] 50 requests: {hit}/{total} prompt tokens from cache ({hit / total:.1%})")
    assert hit == 49 * (2000 // BLOCK) * BLOCK

    traces = []
    for _ in range(1000):
        t0, t = 0.0, rng.expovariate(1 / 0.15)  # first token after queue + prefill, mean 150 ms
        toks = [t]
        for _ in range(rng.randint(50, 300)):
            t += rng.gauss(0.007, 0.001) * (rng.uniform(3, 8) if rng.random() < 0.02 else 1)
            toks.append(t)
        traces.append((t0, toks))
    ttft, tpot, e2e = zip(*map(metrics, traces))
    q = lambda xs: statistics.quantiles(xs, n=100)  # noqa: E731
    print(f"[metrics] TTFT p50 {q(ttft)[49] * 1e3:.0f} ms, p99 {q(ttft)[98] * 1e3:.0f} ms")
    print(f"[metrics] TPOT p50 {q(tpot)[49] * 1e3:.1f} ms, p99 {q(tpot)[98] * 1e3:.1f} ms")
    print(f"[metrics] E2E p50 {q(e2e)[49]:.2f} s")
    for slo in [(0.5, 0.015), (0.3, 0.009), (0.3, 0.008)]:
        print(f"[goodput] TTFT<={slo[0] * 1e3:.0f} ms and TPOT<={slo[1] * 1e3:.0f} ms: {goodput(traces, *slo):.1%}")
