# Adapted from AI Engineering from Scratch (MIT), phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf
"""TF-IDF from scratch (tokenize, vocabulary, counts, DF, IDF, L2 rows, cosine), checked against scikit-learn.

The corpus is the lesson's three-document toy corpus (hand-written, SYNTHETIC). Offline and deterministic.
Run from guide/:  uv run code/09-classical-nlp/tfidf_scratch.py
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Tokenize: words (with an optional apostrophe part) and numbers; punctuation is dropped.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


# --- 2. Vocabulary (sorted, like scikit-learn) and the document-term count matrix [N, V].
def build_vocab(docs):
    return {w: j for j, w in enumerate(sorted({t for doc in docs for t in doc}))}


def count_matrix(docs, vocab):
    X = np.zeros((len(docs), len(vocab)))
    for i, doc in enumerate(docs):
        for t in doc:
            if t in vocab:                    # words unseen at fit time are silently dropped
                X[i, vocab[t]] += 1
    return X


# --- 3. Document frequency and IDF. smooth=False is the course formula ln(N/DF);
#        smooth=True is scikit-learn's default ln((1+N)/(1+DF)) + 1.
def idf(X, smooth=False):
    N = X.shape[0]
    df = (X > 0).sum(axis=0)                  # [V] number of documents containing each term
    if smooth:
        return np.log((1 + N) / (1 + df)) + 1
    return np.log(N / df)


# --- 4. TF-IDF rows, L2-normalized so cosine similarity is a plain dot product.
def tfidf(X, idf_vec):
    W = X * idf_vec                           # [N, V] broadcast: column j scaled by idf[j]
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.where(norms == 0, 1, norms)


def search(query, vocab, idf_vec, D):
    q = tfidf(count_matrix([tokenize(query)], vocab), idf_vec)[0]
    return D @ q                              # [N] cosine scores (both sides unit length)


if __name__ == "__main__":
    corpus = ["The model learns from data.", "Gradient descent trains the model.", "The cat sat on the mat."]
    docs = [tokenize(d) for d in corpus]
    print("tokens doc 2:", docs[1])
    vocab = build_vocab(docs)
    X = count_matrix(docs, vocab)
    print("vocab:", list(vocab))

    raw_idf = idf(X)
    for w in ["the", "model", "gradient"]:
        j = vocab[w]
        print(f"  '{w}': TF in doc 2 = {X[1, j]:.0f}, DF = {(X[:, j] > 0).sum()}, "
              f"IDF = {raw_idf[j]:.3f}, TF-IDF = {X[1, j] * raw_idf[j]:.3f}")

    # Smoothed IDF + L2 rows should reproduce scikit-learn exactly.
    D = tfidf(X, idf(X, smooth=True))
    sk = TfidfVectorizer(token_pattern=TOKEN_RE.pattern).fit(corpus)
    D_sk = sk.transform(corpus).toarray()
    assert list(sk.get_feature_names_out()) == list(vocab)
    print("max |scratch - sklearn| =", f"{np.abs(D - D_sk).max():.2e}")
    assert np.allclose(D, D_sk)

    print("cosine similarity matrix:\n", (D @ D.T).round(3))
    scores = search("how does gradient descent train a model", vocab, idf(X, smooth=True), D)
    print("query scores:", scores.round(3), "-> best doc", int(scores.argmax()) + 1)
