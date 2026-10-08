"""A tiny RAG retriever (chunking + TF-IDF + a lexical reranker) and retrieval/eval metrics: recall@k, MRR, nDCG.

Everything is offline and deterministic. The document collection is SYNTHETIC.
Run from guide/:  uv run code/16-rag/rag_eval.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

DOCS = {
    "refunds": "Refund policy. Customers may request a refund within 30 days of purchase. Refunds are issued to the original "
               "payment method within 5 business days. Digital downloads are refundable only if they have not been accessed. "
               "Shipping fees are not refundable unless the item arrived damaged.",
    "shipping": "Shipping. Standard shipping takes 3 to 7 business days. Express shipping takes 1 to 2 business days and costs "
                "15 dollars. International orders may take up to 21 days and can incur customs fees paid by the customer.",
    "accounts": "Accounts. Passwords must be at least 12 characters. Two-factor authentication can be enabled in settings. "
                "Accounts inactive for 24 months are closed after an email warning. Deleted accounts cannot be recovered.",
    "warranty": "Warranty. Hardware products include a 1-year limited warranty covering manufacturing defects. The warranty "
                "does not cover accidental damage, water damage, or unauthorized repairs. Extended warranties add 2 years.",
}
QUERIES = [
    ("How many days do I have to ask for my money back?", "refunds", "within 30 days"),
    ("How long does express delivery take?", "shipping", "1 to 2 business days"),
    ("Does the warranty cover water damage?", "warranty", "does not cover"),
    ("What is the minimum password length?", "accounts", "12 characters"),
    ("Are shipping costs refunded if my item was broken on arrival?", "refunds", "arrived damaged"),
]


def chunk(text, size, overlap):
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def build(size, overlap):
    chunks, owner = [], []
    for name, text in DOCS.items():
        for c in chunk(text, size, overlap):
            chunks.append(c)
            owner.append(name)
    vec = TfidfVectorizer(stop_words="english").fit(chunks)
    return chunks, owner, vec, vec.transform(chunks)


def retrieve(q, chunks, vec, M, k):
    s = cosine_similarity(vec.transform([q]), M)[0]
    idx = np.argsort(-s)[:k]
    return idx, s[idx]


def hit(answer_span, retrieved_chunks):
    """Does any retrieved chunk contain the exact evidence needed to answer?"""
    return any(answer_span in c for c in retrieved_chunks)


# ---------------------------------------------------------------------------
# 1. One query, step by step
# ---------------------------------------------------------------------------
chunks, owner, vec, M = build(size=20, overlap=5)
print(f"{len(chunks)} chunks of 20 words (overlap 5)")
q, gold, span = QUERIES[4]
idx, scores = retrieve(q, chunks, vec, M, 3)
print(f"\nQuery: {q}")
for rank, (i, s) in enumerate(zip(idx, scores), 1):
    print(f"  {rank}. [{owner[i]}] score={s:.3f} evidence={'YES' if span in chunks[i] else 'no '} | {chunks[i][:70]}...")

# ---------------------------------------------------------------------------
# 2. Evidence recall@3 across chunk sizes
# ---------------------------------------------------------------------------
sizes = [8, 12, 20, 30, 50, 80]
recalls, avg_len = [], []
for size in sizes:
    chunks, owner, vec, M = build(size, overlap=min(5, size // 3))
    hits = [hit(span, [chunks[i] for i in retrieve(q, chunks, vec, M, 3)[0]]) for q, _, span in QUERIES]
    recalls.append(np.mean(hits))
    avg_len.append(size * 3)
    print(f"chunk size {size:3d}: evidence recall@3 = {np.mean(hits):.2f}   context words sent to the model ≈ {size * 3}")

# A retrieval miss: query words don't overlap the evidence (vocabulary mismatch)
chunks, owner, vec, M = build(20, 5)
miss_q = "Can I get reimbursed for a game I never opened?"
idx, scores = retrieve(miss_q, chunks, vec, M, 3)
print(f"\nMiss example: {miss_q!r}")
print("  top chunks:", [(owner[i], round(float(s), 3)) for i, s in zip(idx, scores)], "(the answer is 'Digital downloads are refundable only if they have not been accessed')")

# ---------------------------------------------------------------------------
# 3. Ranking metrics worked example
# ---------------------------------------------------------------------------
rels = [0, 1, 0, 1, 0]       # relevance of results at ranks 1..5 (1 = relevant), 2 relevant docs exist in total
total_relevant = 2
for k in (1, 3, 5):
    print(f"recall@{k} = {sum(rels[:k]) / total_relevant:.2f}  precision@{k} = {sum(rels[:k]) / k:.2f}")
first = rels.index(1) + 1
print(f"reciprocal rank = 1/{first} = {1 / first:.3f}")
dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)))
print(f"DCG@5 = 1/log2(3) + 1/log2(5) = {dcg:.4f}; IDCG@5 = 1/log2(2) + 1/log2(3) = {idcg:.4f}; nDCG@5 = {dcg / idcg:.4f}")
rrs = [1, 1 / 2, 0, 1 / 3]
print(f"MRR over 4 queries with first relevant at ranks 1, 2, none, 3 = {np.mean(rrs):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
ax = axes[0]
ax.plot(sizes, recalls, "-o", color=TEAL, lw=2, label="evidence recall@3")
ax.set(title="Chunk size trades recall against context length (synthetic docs)", xlabel="chunk size (words)", ylabel="evidence recall@3", ylim=(0, 1.05))
ax2 = ax.twinx()
ax2.plot(sizes, avg_len, "--s", color=ORANGE, label="context words for 3 chunks")
ax2.set_ylabel("context words")
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.legend(handles=ax.lines + ax2.lines, frameon=False, fontsize=8, loc="lower right")
ax.title.set_fontsize(10)
ax = axes[1]
ax.bar(range(1, 6), rels, color=[PURPLE if r else "#d4d4d8" for r in rels])
for i, r in enumerate(rels):
    ax.text(i + 1, r + 0.03, f"gain {r}/log₂({i + 2})\n= {r / np.log2(i + 2):.3f}", ha="center", fontsize=8)
ax.set(title=f"Ranked results: recall@3 = 0.5, RR = 0.5, nDCG@5 = {dcg / idcg:.3f}", xlabel="rank", ylabel="relevant?", ylim=(0, 1.5), yticks=[0, 1])
ax.title.set_fontsize(10)
save(fig, "rag-pipeline")
