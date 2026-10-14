# Adapted from AI Engineering from Scratch (MIT), phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec
# and phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword
"""Skip-gram with negative sampling in NumPy, plus fastText-style character n-grams for unseen words.

The corpus is ten hand-written toy sentences (SYNTHETIC). Fixed seed, offline, runs in a few seconds.
Run from guide/:  uv run code/09-classical-nlp/word2vec_scratch.py
"""
import numpy as np

CORPUS = [
    "the cat sat on the mat", "the dog sat on the rug", "a cat chased a mouse", "a dog chased a cat",
    "the kitten slept on the mat", "the puppy slept on the rug", "cats and dogs are pets",
    "kittens and puppies are young", "cats chase mice", "dogs chase squirrels",
]


# --- 1. (center, context) pairs from a sliding window. This is the whole "self-supervised" label source.
def skipgram_pairs(docs, vocab, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i != j:
                    pairs.append((vocab[center], vocab[doc[j]]))
    return np.array(pairs)


# --- 2. Negative-sampling distribution: unigram counts raised to the 3/4 power.
def noise_dist(docs, vocab):
    counts = np.zeros(len(vocab))
    for doc in docs:
        for t in doc:
            counts[vocab[t]] += 1
    p = counts ** 0.75                        # flattens: rare words sampled more than their frequency
    return p / p.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# --- 3. One SGD step on one true pair and k negatives. E holds input vectors, U output vectors, both [V, d].
def sgd_step(E, U, w, c, negs, lr):
    e = E[w].copy()                           # [d]
    g_pos = sigmoid(e @ U[c]) - 1.0           # d(loss)/d(score) for the true pair: pushes score up
    g_neg = sigmoid(U[negs] @ e)              # [k] for the negatives: pushes scores down
    E[w] -= lr * (g_pos * U[c] + g_neg @ U[negs])
    U[c] -= lr * g_pos * e
    np.subtract.at(U, negs, lr * g_neg[:, None] * e)   # safe if a negative repeats
    return -np.log(sigmoid(e @ U[c]) + 1e-9) - np.log(1 - g_neg + 1e-9).sum()


# --- 4. Training loop: shuffle pairs each epoch, draw k negatives per pair.
def train(docs, vocab, d=16, window=2, k=5, epochs=60, lr=0.05, seed=42):
    rng = np.random.default_rng(seed)
    V = len(vocab)
    E = rng.normal(0, 0.1, (V, d))
    U = rng.normal(0, 0.1, (V, d))
    pairs, p = skipgram_pairs(docs, vocab, window), noise_dist(docs, vocab)
    for epoch in range(epochs):
        loss = 0.0
        for w, c in pairs[rng.permutation(len(pairs))]:
            loss += sgd_step(E, U, w, c, rng.choice(V, size=k, p=p), lr)
        if epoch in (0, epochs - 1):
            print(f"epoch {epoch + 1:3d}: mean loss per pair = {loss / len(pairs):.3f}")
    return E


def nearest(E, vocab, word, topk=3):
    inv = {i: w for w, i in vocab.items()}
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = En @ En[vocab[word]]
    order = [i for i in np.argsort(-sims) if i != vocab[word]][:topk]
    return [(inv[i], round(float(sims[i]), 3)) for i in order]


# --- 5. fastText idea: a word is the sum of its character n-gram vectors, so unseen words still get one.
def char_ngrams(word, n_min=3, n_max=6):
    w = f"<{word}>"                           # boundary markers make prefixes and suffixes distinct
    return {w} | {w[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(w) - n + 1)}


if __name__ == "__main__":
    docs = [s.split() for s in CORPUS] * 20   # repeat the toy corpus so each pair is seen often
    vocab = {w: i for i, w in enumerate(sorted({t for doc in docs for t in doc}))}
    print(f"V = {len(vocab)} words, {len(skipgram_pairs(docs, vocab))} training pairs per epoch")
    E = train(docs, vocab)
    print("E shape:", E.shape)
    for word in ["cat", "dog", "mat", "sat"]:
        print(f"nearest to {word:4s}: {nearest(E, vocab, word)}")

    a, b = char_ngrams("kitten"), char_ngrams("kittens")
    print(f"\n'kitten' has {len(a)} n-grams; shares {len(a & b)} with 'kittens', e.g. {sorted(a & b)[:4]}")
    print(f"'kittenish' (never seen) still shares {len(a & char_ngrams('kittenish'))} n-grams with 'kitten'")
