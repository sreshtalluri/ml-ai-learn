# Adapted from AI Engineering from Scratch (MIT), phases/05-nlp-foundations-to-advanced/15-topic-modeling
"""LDA topic modeling: collapsed Gibbs sampling from scratch in NumPy, then scikit-learn's LDA and NMF.

The corpus is SYNTHETIC: 60 short documents generated from three hand-written word lists
(finance, AI research, politics), some of them mixing two topics.
Fixed seed, offline, runs in a few seconds.
Run from guide/:  uv run code/09-classical-nlp/topic_model_scratch.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, TEAL, plt, save  # noqa: E402

rng = np.random.default_rng(0)

TRUE_TOPICS = {
    "finance": "stocks bonds rates yields investors market earnings fed inflation treasury".split(),
    "ai": "model training gpu attention dataset benchmark neural inference tokens researchers".split(),
    "politics": "senate bill vote congress president tariffs election policy governor campaign".split(),
}


def make_corpus(n_docs=60, doc_len=30):
    """Each doc: a mixture over the 3 true topics (pure or two-topic), words drawn from that topic's list."""
    names = list(TRUE_TOPICS)
    docs, mixes = [], []
    for d in range(n_docs):
        mix = np.zeros(3)
        a = d % 3
        if d % 4 == 3:                                   # every 4th doc mixes two topics 60/40
            mix[a], mix[(a + 1) % 3] = 0.6, 0.4
        else:
            mix[a] = 1.0
        topics = rng.choice(3, size=doc_len, p=mix)
        docs.append([str(rng.choice(TRUE_TOPICS[names[k]])) for k in topics])
        mixes.append(mix)
    return docs, np.array(mixes)


# ---------------------------------------------------------------------------
# Collapsed Gibbs sampling for LDA
# ---------------------------------------------------------------------------
def gibbs_lda(docs, K, alpha=0.1, beta=0.01, iters=200):
    vocab = sorted({w for d in docs for w in d})
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    W = [np.array([w2i[w] for w in d]) for d in docs]
    Z = [rng.integers(0, K, len(d)) for d in W]           # random initial topic per token
    ndk = np.zeros((len(W), K))                           # tokens in doc d assigned to topic k
    nkw = np.zeros((K, V))                                # times word w is assigned to topic k
    nk = np.zeros(K)                                      # tokens assigned to topic k
    for d, (ws, zs) in enumerate(zip(W, Z)):
        for w, z in zip(ws, zs):
            ndk[d, z] += 1
            nkw[z, w] += 1
            nk[z] += 1

    for _ in range(iters):
        for d, (ws, zs) in enumerate(zip(W, Z)):
            for i, (w, z) in enumerate(zip(ws, zs)):
                ndk[d, z] -= 1                            # remove this token's current assignment
                nkw[z, w] -= 1
                nk[z] -= 1
                p = (ndk[d] + alpha) * (nkw[:, w] + beta) / (nk + V * beta)   # [K]
                z = rng.choice(K, p=p / p.sum())          # resample its topic
                zs[i] = z
                ndk[d, z] += 1
                nkw[z, w] += 1
                nk[z] += 1

    theta = (ndk + alpha) / (ndk.sum(1, keepdims=True) + K * alpha)        # [D, K] doc-topic
    phi = (nkw + beta) / (nkw.sum(1, keepdims=True) + V * beta)            # [K, V] topic-word
    return theta, phi, vocab


def top_words(phi, vocab, n=6):
    return [[vocab[i] for i in np.argsort(-row)[:n]] for row in phi]


def topic_diversity(tops):
    """Fraction of unique words across all topics' top-n lists (1.0 = no overlap)."""
    flat = [w for t in tops for w in t]
    return len(set(flat)) / len(flat)


def main():
    print("== Worked example: one Gibbs update ==")
    # word 'model' in a doc; counts with the token itself removed; K = 2, V = 30
    ndk, nkw, nk, a, b, V = np.array([7, 2]), np.array([0, 9]), np.array([150, 140]), 0.1, 0.01, 30
    p = (ndk + a) * (nkw + b) / (nk + V * b)
    print(f"unnormalised: topic0 = (7.1)(0.01)/150.3 = {p[0]:.6f}, topic1 = (2.1)(9.01)/140.3 = {p[1]:.6f}")
    print(f"normalised:   P(z=0) = {p[0] / p.sum():.3f}, P(z=1) = {p[1] / p.sum():.3f}")

    docs, mixes = make_corpus()
    print(f"\nSYNTHETIC corpus: {len(docs)} docs, {sum(map(len, docs))} tokens, "
          f"vocabulary {len({w for d in docs for w in d})}")

    theta, phi, vocab = gibbs_lda(docs, K=3)
    tops = top_words(phi, vocab)
    print("\n== From-scratch collapsed Gibbs LDA, K = 3, 200 sweeps ==")
    for k, t in enumerate(tops):
        print(f"topic {k}: {' '.join(t)}")
    print(f"topic diversity (top 6): {topic_diversity(tops):.2f}")
    for d in [0, 3]:
        print(f"doc {d}: true mix {mixes[d]}, inferred theta {np.round(theta[d], 2)}")

    print("\n== scikit-learn LDA and NMF on the same counts ==")
    texts = [" ".join(d) for d in docs]
    cv = CountVectorizer()
    X = cv.fit_transform(texts)                        # LDA wants raw counts, not TF-IDF
    names = cv.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=3, random_state=0).fit(X)
    for k, row in enumerate(lda.components_):
        print(f"sklearn LDA topic {k}: {' '.join(names[np.argsort(-row)[:6]])}")
    nmf = NMF(n_components=3, init="nndsvd", random_state=0).fit(X)
    for k, row in enumerate(nmf.components_):
        print(f"sklearn NMF topic {k}: {' '.join(names[np.argsort(-row)[:6]])}")

    print("\n== Choosing K: diversity of top-6 words (from-scratch LDA) ==")
    div = {}
    for K in [2, 3, 5, 8]:
        _, phi_k, _ = gibbs_lda(docs, K=K, iters=100)
        div[K] = topic_diversity(top_words(phi_k, vocab))
        print(f"K = {K}: diversity {div[K]:.2f}")

    # ---------------------------- figure ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    for k, col in zip(range(3), [BLUE, ORANGE, TEAL]):
        idx = np.argsort(-phi[k])[:6]
        y = np.arange(6) + k * 7
        ax[0].barh(y, phi[k, idx], color=col)
        for yy, i in zip(y, idx):
            ax[0].text(phi[k, i] + 0.003, yy, vocab[i], va="center", fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_yticks([2.5, 9.5, 16.5], ["topic 0", "topic 1", "topic 2"])
    ax[0].set(title="Topic-word distributions (top 6 words)", xlabel="P(word | topic)", xlim=(0, phi.max() * 1.3))
    show = list(range(12))
    im = ax[1].imshow(theta[show], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax[1].set_yticks(show, [f"doc {d}" + (" (mixed)" if mixes[d].max() < 1 else "") for d in show], fontsize=8)
    ax[1].set_xticks(range(3), ["topic 0", "topic 1", "topic 2"])
    ax[1].set_title("Doc-topic mixtures (first 12 docs)")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    save(fig, "topic-modeling")


if __name__ == "__main__":
    main()
