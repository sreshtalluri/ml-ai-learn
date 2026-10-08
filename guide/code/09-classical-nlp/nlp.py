"""Bag-of-words, n-grams, TF-IDF, cosine similarity, and count-based word embeddings (PPMI + SVD).

Run from guide/:  uv run code/09-classical-nlp/nlp.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Bag-of-words and bigrams
# ---------------------------------------------------------------------------
docs = ["machine learning is useful", "machine learning learning"]
cv = CountVectorizer()
print("vocab:", cv.fit(docs).get_feature_names_out().tolist())
print("BoW:\n", cv.transform(docs).toarray())
bi = CountVectorizer(ngram_range=(1, 2)).fit(["the movie was not good", "the movie was good"])
print("with bigrams:", bi.get_feature_names_out().tolist())

# ---------------------------------------------------------------------------
# 2. TF-IDF by hand (course formula: raw count x ln(N / df))
# ---------------------------------------------------------------------------
print(f"\nCourse example: N=100, tf=3, df=5 -> idf = ln(20) = {np.log(20):.4f}, tf-idf = {3 * np.log(20):.4f}")
corpus = ["the model learns from data", "gradient descent trains the model", "the cat sat on the mat"]
tokens = [d.split() for d in corpus]
N = len(corpus)
for word in ["model", "the", "gradient"]:
    df = sum(word in t for t in tokens)
    tf = tokens[1].count(word)
    print(f"  '{word}' in doc 2: tf={tf}, df={df}, idf=ln({N}/{df})={np.log(N / df):.4f}, tf-idf={tf * np.log(N / df):.4f}")
# scikit-learn's default is smoothed and L2-normalized, so its numbers differ:
tv = TfidfVectorizer(smooth_idf=True, norm="l2").fit(corpus)
print("  sklearn idf (smoothed: ln((1+N)/(1+df)) + 1):", dict(zip(tv.get_feature_names_out(), tv.idf_.round(3))))
X = tv.transform(corpus)
print("  cosine similarity matrix:\n", cosine_similarity(X).round(3))

a, b = np.array([1, 2, 0]), np.array([2, 1, 1])
print(f"\ncosine([1,2,0],[2,1,1]) = {a @ b} / ({np.linalg.norm(a):.4f} * {np.linalg.norm(b):.4f}) = {a @ b / np.linalg.norm(a) / np.linalg.norm(b):.4f}")

# ---------------------------------------------------------------------------
# 3. Count-based embeddings: co-occurrence -> PPMI -> SVD (SYNTHETIC corpus)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
groups = {
    "animals": ["cat", "dog", "puppy", "kitten"],
    "royalty": ["king", "queen", "prince", "princess"],
    "food": ["pizza", "pasta", "bread", "cheese"],
}
context = {"animals": ["pet", "fur", "vet", "play"], "royalty": ["crown", "palace", "throne", "royal"], "food": ["eat", "tasty", "cook", "dinner"]}
sentences = []
for _ in range(3000):
    g = rng.choice(list(groups))
    sentences.append([rng.choice(groups[g]), rng.choice(context[g]), rng.choice(context[g]), rng.choice(sum(context.values(), []))])
vocab = sorted({w for s in sentences for w in s})
ix = {w: i for i, w in enumerate(vocab)}
co = np.zeros((len(vocab), len(vocab)))
for s in sentences:
    for u, v in combinations(s, 2):
        co[ix[u], ix[v]] += 1
        co[ix[v], ix[u]] += 1
total = co.sum()
pw = co.sum(1) / total
with np.errstate(divide="ignore"):
    pmi = np.log((co / total) / np.outer(pw, pw))
ppmi = np.maximum(pmi, 0)
U, S, _ = np.linalg.svd(ppmi)
emb = U[:, :2] * S[:2]
words = [w for g in groups.values() for w in g]
E = emb[[ix[w] for w in words]]
En = E / np.linalg.norm(E, axis=1, keepdims=True)
sim = En @ En.T
print("\nembedding cosine: cat~dog {:.3f}, cat~king {:.3f}, pizza~pasta {:.3f}".format(
    sim[words.index("cat"), words.index("dog")], sim[words.index("cat"), words.index("king")], sim[words.index("pizza"), words.index("pasta")]))
# TF-IDF can't see that 'cat' and 'kitten' are related if they never co-occur in a document:
t2 = TfidfVectorizer().fit_transform(["my cat sleeps", "a kitten sleeps", "the king rules"])
print("TF-IDF cosine 'my cat sleeps' vs 'a kitten sleeps':", cosine_similarity(t2)[0, 1].round(3), "(shares only 'sleeps')")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]
im = ax.imshow(cosine_similarity(X), cmap="Purples", vmin=0, vmax=1)
ax.set_xticks(range(N), [f"doc {i + 1}" for i in range(N)])
ax.set_yticks(range(N), [c[:28] for c in corpus])
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{cosine_similarity(X)[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set(title="TF-IDF cosine similarity between documents")
ax.grid(False)
ax = axes[1]
cols = {"animals": BLUE, "royalty": PURPLE, "food": ORANGE}
for g, ws in groups.items():
    pts = emb[[ix[w] for w in ws]]
    ax.scatter(*pts.T, color=cols[g], s=40, label=g)
    for w, p in zip(ws, pts):
        ax.annotate(w, p, xytext=(4, 3), textcoords="offset points", fontsize=9)
ax.set(title="Count-based word embeddings (PPMI + SVD), synthetic corpus", xlabel="dim 1", ylabel="dim 2")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
save(fig, "text-to-vectors")

fig, ax = plt.subplots(figsize=(6, 4.2))
for g, ws in groups.items():
    pts = emb[[ix[w] for w in ws]]
    ax.scatter(*pts.T, color=cols[g], s=50, label=g)
    for w, p in zip(ws, pts):
        ax.annotate(w, p, xytext=(4, 3), textcoords="offset points", fontsize=9)
ctx_words = sum(context.values(), [])
cp = emb[[ix[w] for w in ctx_words]]
ax.scatter(*cp.T, color=TEAL, s=15, marker="^", label="context words")
ax.set(title="Words used in similar contexts land close together", xlabel="dim 1", ylabel="dim 2")
ax.title.set_fontsize(10)
ax.legend(frameon=False, fontsize=8)
save(fig, "word-embeddings")
