# Adapted from AI Engineering from Scratch (MIT), phases/05-nlp-foundations-to-advanced/14-information-retrieval-search, phases/11-llm-engineering/04-embeddings, phases/19-capstone-projects/65-hybrid-retrieval-bm25-dense
"""Vector search from scratch: similarity metrics, exact top-k, an IVF index on our own k-means,
product quantization with lookup-table distances, BM25, and reciprocal rank fusion.

Synthetic: clustered random vectors stand in for embeddings; the BM25 corpus is made up.
Run from guide/:  uv run code/16-rag/vector_search_scratch.py
"""
import math
import re
from collections import Counter

import numpy as np


# --- 1. Exact search: normalize once, then dot product --------------------------
def normalize(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def exact_topk(X, q, k):                          # X: [N, d] normalized, q: [d] normalized
    s = X @ q                                     # one matrix-vector product: N dot products
    top = np.argpartition(-s, k)[:k]              # top k in O(N), unsorted
    return top[np.argsort(-s[top])]


# --- 2. k-means: the building block of IVF and PQ --------------------------------
def kmeans(X, k, rng, iters=10):
    C = X[rng.choice(len(X), k, replace=False)]   # start from k random points
    for _ in range(iters):
        d2 = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)   # squared distances [N, k]
        lab = d2.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C, lab


# --- 3. IVF: scan only the cells nearest the query -------------------------------
class IVF:
    def __init__(self, X, nlist, rng):
        self.X = X
        self.C, lab = kmeans(X, nlist, rng)
        self.lists = [np.flatnonzero(lab == c) for c in range(nlist)]   # inverted lists

    def search(self, q, k, nprobe):
        cells = np.argsort(-(self.C @ q))[:nprobe]                     # nearest centroids
        cand = np.concatenate([self.lists[c] for c in cells])
        return cand[np.argsort(-(self.X[cand] @ q))[:k]], len(cand)


# --- 4. Product quantization: 1 byte per subvector -------------------------------
class PQ:
    def __init__(self, X, m, rng):
        self.m, self.ds = m, X.shape[1] // m      # m subvectors of ds dimensions
        self.books, codes = [], []
        for j in range(m):
            sub = X[:, j * self.ds:(j + 1) * self.ds]
            C, lab = kmeans(sub, 256, rng)        # 256 centroids -> a code fits in one byte
            self.books.append(C); codes.append(lab)
        self.codes = np.stack(codes, 1).astype(np.uint8)   # [N, m]: the whole stored index

    def search(self, q, k):
        table = np.stack([self.books[j] @ q[j * self.ds:(j + 1) * self.ds] for j in range(self.m)])  # [m, 256]
        s = table[np.arange(self.m), self.codes].sum(1)    # m lookups per vector, no float math on X
        return np.argsort(-s)[:k]


# --- 5. BM25 and reciprocal rank fusion ------------------------------------------
def bm25_scores(docs, query, k1=1.5, b=0.75):
    toks = [re.findall(r"\w+", d.lower()) for d in docs]
    avgdl = sum(map(len, toks)) / len(toks)
    df = Counter(t for d in toks for t in set(d))
    scores = []
    for d in toks:
        tf, s = Counter(d), 0.0
        for t in re.findall(r"\w+", query.lower()):
            if t in tf:
                idf = math.log((len(toks) - df[t] + 0.5) / (df[t] + 0.5) + 1)
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def rrf(rankings, c=60):
    s = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            s[doc] = s.get(doc, 0.0) + 1 / (c + rank)
    return sorted(s, key=s.get, reverse=True), s


if __name__ == "__main__":
    print("== Metrics worked example: q=[1,0], a=[0.8,0.6], b=[3,4] ==")
    q, a, b = np.array([1.0, 0]), np.array([0.8, 0.6]), np.array([3.0, 4])
    print(f"dot a={q @ a:.1f} b={q @ b:.1f} | cos a={q @ a / np.linalg.norm(a):.1f} b={q @ b / np.linalg.norm(b):.1f}"
          f" | L2 a={np.linalg.norm(q - a):.3f} b={np.linalg.norm(q - b):.3f}")

    rng = np.random.default_rng(0)
    centers = rng.normal(0, 1, (64, 32))
    X = normalize(centers[rng.integers(0, 64, 20_000)] + 0.35 * rng.normal(0, 1, (20_000, 32)))
    Q = normalize(centers[rng.integers(0, 64, 100)] + 0.35 * rng.normal(0, 1, (100, 32)))
    truth = [set(exact_topk(X, qq, 10)) for qq in Q]
    recall = lambda found: np.mean([len(set(f) & t) / 10 for f, t in zip(found, truth)])  # noqa: E731

    print("\n== IVF on SYNTHETIC 32-d data (N = 20,000, nlist = 128, 100 queries) ==")
    ivf = IVF(X, 128, rng)
    for nprobe in (1, 4, 16):
        res = [ivf.search(qq, 10, nprobe) for qq in Q]
        print(f"nprobe={nprobe:2d}: recall@10 {recall([r[0] for r in res]):.3f}, "
              f"vectors scanned {np.mean([r[1] for r in res]) / len(X):.1%} of N")

    print("\n== Product quantization, m = 16 subvectors of 2 dims ==")
    pq = PQ(X, 16, rng)
    print(f"memory: float32 {X.shape[1] * 4} bytes/vector -> PQ {pq.codes.shape[1]} bytes/vector ({X.shape[1] * 4 // pq.codes.shape[1]}x smaller)")
    pq_only = [pq.search(qq, 10) for qq in Q]
    refined = []
    for qq in Q:                                  # PQ shortlist of 100, then exact rerank on those 100
        cand = pq.search(qq, 100)
        refined.append(cand[np.argsort(-(X[cand] @ qq))[:10]])
    print(f"recall@10 PQ only {recall(pq_only):.3f}; PQ top-100 + exact rerank {recall(refined):.3f}")

    print("\n== BM25 (synthetic snippets) ==")
    docs = ["error E1234 when syncing calendar", "calendar sync fails after update",
            "how to export contacts", "sync settings and sync frequency for calendar"]
    print("query 'E1234':", [round(s, 3) for s in bm25_scores(docs, "E1234")], "<- exact IDs are BM25's strength")

    print("\n== RRF worked example: dense A,B,C,D; BM25 E,F,A,B ==")
    order, s = rrf([list("ABCD"), list("EFAB")])
    print("fused:", order, {d: round(v, 6) for d, v in s.items()})
