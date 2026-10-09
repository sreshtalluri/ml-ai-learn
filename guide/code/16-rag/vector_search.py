"""Exact vs approximate nearest-neighbor search: an IVF index from scratch, recall@k against brute force,
cost and memory arithmetic, and reciprocal rank fusion.

All vectors are SYNTHETIC (Gaussian clusters, fixed seeds). Offline and deterministic.
Run from guide/:  uv run code/16-rag/vector_search.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def synthetic(n, d, n_clusters, seed, spread=1.0):
    r = np.random.default_rng(seed)
    centers = r.normal(0, 4, (n_clusters, d))
    return (centers[r.integers(0, n_clusters, n)] + r.normal(0, spread, (n, d))).astype(np.float32)


def brute_force(X, q, k):
    d2 = ((X - q) ** 2).sum(1)
    return np.argsort(d2)[:k]


class IVF:
    """Inverted file index: k-means partitions the vectors; a query scans only the nprobe nearest cells."""

    def __init__(self, X, nlist, seed=0):
        km = KMeans(n_clusters=nlist, n_init=1, random_state=seed).fit(X)
        self.X, self.centroids = X, km.cluster_centers_
        self.lists = [np.flatnonzero(km.labels_ == c) for c in range(nlist)]

    def search(self, q, k, nprobe):
        cells = np.argsort(((self.centroids - q) ** 2).sum(1))[:nprobe]
        cand = np.concatenate([self.lists[c] for c in cells])
        d2 = ((self.X[cand] - q) ** 2).sum(1)
        top = cand[np.argsort(d2)[:k]]
        return top, len(self.centroids) + len(cand), cells  # results, distance computations, probed cells


def recall_at_k(found, exact):
    return len(set(found) & set(exact)) / len(exact)


def rrf(rankings, c=60):
    """Reciprocal rank fusion: score(doc) = sum over rankers of 1 / (c + rank), rank starting at 1."""
    s = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            s[doc] = s.get(doc, 0.0) + 1 / (c + rank)
    return sorted(s.items(), key=lambda kv: -kv[1])


if __name__ == "__main__":
    print("== Cost arithmetic: N = 1,000,000 vectors, d = 768, IVF nlist = 1024, nprobe = 8 ==")
    N, d, nlist, nprobe = 1_000_000, 768, 1024, 8
    brute = N * d
    scanned = nprobe / nlist * N
    ivf = (nlist + scanned) * d
    print(f"brute force: {N:,} distances x {d} = {brute:,} multiply-adds")
    print(f"IVF: {nlist} centroid distances + {nprobe}/{nlist} x {N:,} = {scanned:,.1f} vectors scanned")
    print(f"     {nlist + scanned:,.1f} distances x {d} = {ivf:,.0f} multiply-adds ({100 * ivf / brute:.2f}%, {brute / ivf:.0f}x fewer)")
    print(f"float32 vectors: {N * d * 4 / 1e9:.3f} GB; float16: {N * d * 2 / 1e9:.3f} GB; PQ 96 bytes: {N * 96 / 1e6:.0f} MB")
    print(f"HNSW layer-0 links, M = 32 (2M = 64 neighbors x 4-byte ids): {N * 64 * 4 / 1e6:.0f} MB")

    print("\n== recall@10 example ==")
    exact = list(range(10))
    found = [0, 1, 2, 3, 4, 5, 6, 7, 42, 99]
    print(f"exact top-10 {exact}\nANN top-10   {found}\nrecall@10 = {recall_at_k(found, exact):.1f}")

    print("\n== Reciprocal rank fusion (c = 60) ==")
    dense, bm25 = ["A", "B", "C", "D"], ["E", "F", "A", "B"]
    for doc, score in rrf([dense, bm25]):
        print(f"  {doc}: {score:.6f}")

    print("\n== IVF recall vs nprobe on SYNTHETIC 64-d data (N = 20,000, nlist = 128, 200 held-out queries) ==")
    data = synthetic(20_200, 64, 60, seed=7, spread=6.0)
    X, Q = data[:20_000], data[20_000:]
    index = IVF(X, 128, seed=0)
    exact_all = [brute_force(X, q, 10) for q in Q]
    probes = [1, 2, 4, 8, 16, 32, 64, 128]
    rec, frac = [], []
    for p in probes:
        rs, cs = [], []
        for q, ex in zip(Q, exact_all):
            top, cost, _ = index.search(q, 10, p)
            rs.append(recall_at_k(top, ex))
            cs.append(cost)
        rec.append(np.mean(rs))
        frac.append(np.mean(cs) / len(X))
        print(f"  nprobe={p:4d}  recall@10={rec[-1]:.3f}  distance computations = {100 * frac[-1]:6.2f}% of brute force")

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    P = synthetic(600, 2, 8, seed=3, spread=1.3)
    small = IVF(P, 16, seed=0)
    q = np.array([-1.1, 6.4], dtype=np.float32)
    top, cost, cells = small.search(q, 10, 2)
    ex = brute_force(P, q, 10)
    probed = np.zeros(len(P), bool)
    for c in cells:
        probed[small.lists[c]] = True
    a1.scatter(*P[~probed].T, s=8, color=GRAY, alpha=0.5, label="not scanned")
    a1.scatter(*P[probed].T, s=10, color=BLUE, alpha=0.7, label="scanned (2 probed cells)")
    a1.scatter(*small.centroids.T, marker="x", color="#27272a", s=40, label="cell centroids")
    missed = [i for i in ex if i not in set(top)]
    a1.scatter(*P[[i for i in ex if i in set(top)]].T, s=60, facecolors="none", edgecolors=TEAL, lw=1.8, label="true top-10, found")
    if missed:
        a1.scatter(*P[missed].T, s=60, facecolors="none", edgecolors=ORANGE, lw=1.8, label="true top-10, missed")
    a1.scatter(*q, marker="*", s=220, color=PURPLE, label="query", zorder=5)
    a1.set_title(f"IVF, nlist = 16, nprobe = 2: recall@10 = {recall_at_k(top, ex):.1f}")
    a1.legend(fontsize=7, loc="lower left")
    a1.set_xlabel("dimension 1")
    a1.set_ylabel("dimension 2")
    print(f"\nfigure query: scanned {cost - 16} of {len(P)} points, recall@10 = {recall_at_k(top, ex):.1f}, missed {len(missed)}")

    a2.plot(np.array(frac) * 100, rec, "o-", color=BLUE)
    for p, f, r in zip(probes, frac, rec):
        a2.annotate(f"nprobe {p}", (f * 100, r), textcoords="offset points", xytext=(6, -12) if p < 64 else (-4, -16), ha="left" if p < 64 else "right", fontsize=8)
    a2.set_xscale("log")
    a2.set_xlabel("distance computations (% of brute force, log scale)")
    a2.set_ylabel("recall@10 vs brute force")
    a2.set_title("Synthetic 64-d data, N = 20,000, nlist = 128")
    a2.set_ylim(0, 1.05)
    save(fig, "vector-search")
