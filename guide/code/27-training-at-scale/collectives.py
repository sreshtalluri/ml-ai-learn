# Adapted from AI Engineering from Scratch (MIT), phases/19-capstone-projects/76-collective-ops-from-scratch
"""Collective operations from scratch: ring reduce-scatter, ring all-gather, ring all-reduce, and a
binomial-tree broadcast, simulated for N ranks in one process. Every rank's sends are counted, so the
2(N-1)/N bandwidth result can be checked, and every result is verified against np.sum.

Tensors and link speeds are SYNTHETIC. Deterministic.
Run from guide/:  uv run code/27-training-at-scale/collectives.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, plt, save  # noqa: E402


class Ring:
    """N ranks on a ring. A step is synchronous: every rank sends one chunk to its right neighbour."""

    def __init__(self, n):
        self.n = n
        self.sent = [0] * n      # floats sent by each rank
        self.steps = 0

    def shift(self, outgoing):
        """outgoing[r] is what rank r sends to rank r+1. Returns what each rank receives."""
        for r, x in enumerate(outgoing):
            self.sent[r] += x.size
        self.steps += 1
        return [outgoing[(r - 1) % self.n] for r in range(self.n)]


def reduce_scatter(ring, tensors):
    """Ring reduce-scatter. After N-1 steps rank r holds the full sum of chunk (r + 1) mod N."""
    n = ring.n
    chunks = [np.array_split(t.astype(np.float64), n) for t in tensors]   # chunks[rank][chunk]
    for s in range(n - 1):
        send_idx = [(r - s) % n for r in range(n)]
        received = ring.shift([chunks[r][send_idx[r]] for r in range(n)])
        for r in range(n):
            idx = (r - s - 1) % n                       # the chunk my left neighbour just sent
            chunks[r][idx] = chunks[r][idx] + received[r]
    return [((r + 1) % n, chunks[r][(r + 1) % n]) for r in range(n)]    # (chunk index, summed chunk)


def all_gather(ring, owned, n_chunks):
    """Ring all-gather. owned[r] = (chunk index, data). After N-1 steps every rank has every chunk."""
    n = ring.n
    have = [{i: d} for i, d in owned]
    carry = list(owned)
    for _ in range(n - 1):
        ring.shift([d for _, d in carry])               # count the sends
        carry = [carry[(r - 1) % n] for r in range(n)]  # pass (index, data) to the right
        for r in range(n):
            have[r][carry[r][0]] = carry[r][1]
    return [np.concatenate([h[i] for i in range(n_chunks)]) for h in have]


def all_reduce(ring, tensors):
    """Ring all-reduce = reduce-scatter, then all-gather of the summed chunks."""
    return all_gather(ring, reduce_scatter(ring, tensors), ring.n)


def tree_broadcast(n, src_value):
    """Binomial tree: every round, each rank that already has the value sends it to one rank that does not."""
    have, rounds, sends = {0: src_value}, 0, 0
    while len(have) < n:
        for r in list(have):
            partner = r + 2**rounds
            if partner < n:
                have[partner] = have[r]
                sends += 1
        rounds += 1
    return have, rounds, sends


# ---------- alpha-beta cost model: time = messages x alpha + bytes x beta ----------

def t_ring(n, T, alpha, beta):
    return 2 * (n - 1) * alpha + 2 * (n - 1) / n * T * beta


def t_tree(n, T, alpha, beta):
    """Reduce up a binary tree, then broadcast down: 2 log2(N) rounds, each moving the full tensor."""
    return 2 * np.log2(n) * (alpha + T * beta)


def t_server(n, T, alpha, beta):
    """One parameter server receives N-1 tensors and sends N-1 back, one after another, through its single link."""
    return 2 * (n - 1) * (alpha + T * beta)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T = 4, 8
    tensors = [rng.integers(0, 10, T).astype(float) for _ in range(N)]   # SYNTHETIC gradients
    print(f"1) {N} ranks, tensor of {T} floats each")
    for r, t in enumerate(tensors):
        print(f"  rank {r}: {t.astype(int).tolist()}")
    truth = np.sum(tensors, axis=0)
    print(f"  true sum: {truth.astype(int).tolist()}")

    ring = Ring(N)
    rs = reduce_scatter(ring, tensors)
    for r, (i, d) in enumerate(rs):
        print(f"  after reduce-scatter rank {r} holds chunk {i} = {d.astype(int).tolist()}")
    out = all_gather(ring, rs, N)
    print(f"  after all-gather every rank equals the sum: {all(np.array_equal(o, truth) for o in out)}")
    print(f"  steps = {ring.steps} (= 2(N-1)); floats sent per rank = {ring.sent} (2(N-1)/N x T = {2 * (N - 1) / N * T:.0f})")

    print("\n2) Bytes sent per rank by ring all-reduce of a 1 GB tensor")
    for n in (2, 4, 8, 64, 1024):
        print(f"  N = {n:4d}: 2(N-1)/N = {2 * (n - 1) / n:.4f} GB")

    have, rounds, sends = tree_broadcast(8, "w")
    print(f"\n3) Tree broadcast to 8 ranks: {rounds} rounds (log2 8), {sends} sends, all have it: {len(have) == 8}")

    alpha, beta = 10e-6, 1 / 50e9          # SYNTHETIC link: 10 microseconds latency, 50 GB/s
    print("\n4) alpha-beta model, 10 us per message, 50 GB/s per link, N = 64")
    for T_bytes in (4e3, 1e6, 100e6, 1e9):
        r, t, s = (f(64, T_bytes, alpha, beta) * 1e3 for f in (t_ring, t_tree, t_server))
        print(f"  {T_bytes / 1e6:8.3f} MB: ring {r:9.3f} ms  tree {t:9.3f} ms  server {s:9.3f} ms")

    # ---------- figure ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7))
    sizes = np.logspace(3, 9.5, 60)
    for f, col, lab in ((t_ring, BLUE, "ring"), (t_tree, PURPLE, "binary tree"), (t_server, ORANGE, "parameter server")):
        a1.plot(sizes, [f(64, s, alpha, beta) * 1e3 for s in sizes], color=col, label=lab)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("tensor size (bytes)")
    a1.set_ylabel("all-reduce time (ms, model)")
    a1.set_title("N = 64, 10 us latency, 50 GB/s (synthetic)")
    a1.legend(frameon=False, fontsize=9)
    ns = np.array([2, 4, 8, 16, 32, 64, 128])
    a2.plot(ns, 2 * (ns - 1) / ns, "o-", color=BLUE, label="ring: 2(N-1)/N")
    a2.plot(ns, ns - 1, "o-", color=ORANGE, label="parameter server link, each way: N-1")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xticks(ns, [str(n) for n in ns])
    a2.set_xlabel("ranks N")
    a2.set_ylabel("tensor sizes sent (busiest link)")
    a2.set_title("Ring traffic per rank stays under 2x")
    a2.legend(frameon=False, fontsize=9)
    save(fig, "collective-ops-from-scratch")
