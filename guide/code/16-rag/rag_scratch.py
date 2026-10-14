# Adapted from AI Engineering from Scratch (MIT), phases/11-llm-engineering/06-rag, phases/11-llm-engineering/07-advanced-rag, phases/19-capstone-projects/64-chunking-strategies-advanced, phases/19-capstone-projects/66-reranker-cross-encoder, phases/19-capstone-projects/67-query-rewriting-hyde
"""A RAG pipeline from scratch: chunk, index with TF-IDF, retrieve, rewrite the query, rerank,
and build a grounded prompt whose citations you can check.

Same SYNTHETIC documents, stop words, synonym map, and scoring as the website's RAG lab
(web/src/lib/rag.ts) and code/16-rag/rag_eval.py, so all three print the same numbers.
The synonym map and the reranker are hand-written stand-ins for an LLM rewriter and a
cross-encoder model, so everything runs offline.
Run from guide/:  uv run code/16-rag/rag_scratch.py
"""
import math
import re

import numpy as np

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
QUERIES = [  # (question, evidence span that must reach the model)
    ("How many days do I have to ask for my money back?", "within 30 days"),
    ("How long does express delivery take?", "1 to 2 business days"),
    ("Does the warranty cover water damage?", "does not cover"),
    ("Are shipping costs refunded if my item was broken on arrival?", "arrived damaged"),
    ("Can I get reimbursed for a game I never opened?", "not been accessed"),
]
STOP = set("a an the and or of to in on for is are be can if my i do does how what which with it its that this "
           "have has may must not was were by at from as".split())
SYNONYMS = {"reimbursed": ["refund", "refundable"], "money": ["refund"], "back": ["refund"], "game": ["digital", "downloads"],
            "opened": ["accessed"], "never": ["not"], "broken": ["damaged"], "delivery": ["shipping"], "costs": ["fees"],
            "password": ["passwords"]}


def terms(s):
    return [t for t in re.findall(r"[\w']+", s.lower()) if t not in STOP]


# --- 1. Chunking: fixed word windows with overlap, or whole sentences -------------
def fixed_chunks(size, overlap):
    out, step = [], max(1, size - overlap)
    for doc, text in DOCS.items():
        w = text.split()
        out += [(doc, " ".join(w[i:i + size])) for i in range(0, max(1, len(w) - overlap), step)]
    return out                                    # [(source, text)]


def sentence_chunks(max_words):
    out = []
    for doc, text in DOCS.items():
        cur = []
        for sent in re.split(r"(?<=\.)\s+", text):  # never cut a sentence in half
            if cur and len(" ".join(cur + [sent]).split()) > max_words:
                out.append((doc, " ".join(cur))); cur = []
            cur.append(sent)
        out.append((doc, " ".join(cur)))
    return out


# --- 2. Index and retrieve: TF-IDF vectors, cosine similarity ---------------------
def build_index(chunks):
    docs = [terms(t) for _, t in chunks]
    vocab = sorted({t for d in docs for t in d})
    idf = np.array([math.log(len(docs) / sum(v in d for d in docs)) for v in vocab])
    col = {v: j for j, v in enumerate(vocab)}

    def vec(toks):
        x = np.zeros(len(vocab))
        for t in toks:
            if t in col:
                x[col[t]] += 1
        return x * idf                            # raw count x ln(N / DF)
    M = np.stack([vec(d) for d in docs])          # [n_chunks, |vocab|]
    return vec, M


def retrieve(chunks, index, query_terms, k):
    vec, M = index
    q = vec(query_terms)
    s = M @ q / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-12)
    order = sorted((i for i in range(len(chunks)) if s[i] > 0), key=lambda i: (-s[i], i))
    return [(i, float(s[i])) for i in order[:k]]


# --- 3. Query rewriting: add the words the documents actually use ------------------
def rewrite(query):
    return [x for t in terms(query) for x in [t] + SYNONYMS.get(t, [])]   # an LLM rewriter in production


# --- 4. Rerank: read query and chunk together, reward close-together matches -------
def rerank_score(chunk_text, query):
    toks = terms(chunk_text)
    q = set(rewrite(query))
    pos = [i for i, t in enumerate(toks) if t in q]
    if not pos:
        return 0.0
    coverage = len({toks[i] for i in pos}) / max(1, len(terms(query)))
    span = pos[-1] - pos[0] + 1 if len(pos) > 1 else len(toks)
    return round(coverage * (1 + len(pos) / span), 4)


# --- 5. Grounded prompt and a citation check ---------------------------------------
def build_prompt(question, chunks, hits):
    sources = "\n".join(f"[{n}] ({chunks[i][0]}) {chunks[i][1]}" for n, (i, _) in enumerate(hits, 1))
    return ("Answer using ONLY the sources below. Cite sources like [1]. "
            "If the sources do not contain the answer, say you don't know.\n\n"
            f"Sources:\n{sources}\n\nQuestion: {question}")


def check_citations(answer, n_sources):
    cited = {int(c) for c in re.findall(r"\[(\d+)\]", answer)}
    return {"cited": sorted(cited), "invalid": sorted(c for c in cited if not 1 <= c <= n_sources), "uncited": not cited}


def recall_at(chunks, k, use_rewrite=False):
    idx = build_index(chunks)
    hits = [any(span in chunks[i][1] for i, _ in retrieve(chunks, idx, rewrite(q) if use_rewrite else terms(q), k))
            for q, span in QUERIES]
    return sum(hits) / len(hits)


if __name__ == "__main__":
    q, span = QUERIES[3]
    print(f"== Worked example: {q!r} ==")
    for size in (20, 30):
        chunks = fixed_chunks(size, 5)
        hits = retrieve(chunks, build_index(chunks), terms(q), 3)
        print(f"{len(chunks)} chunks of {size} words:", [(chunks[i][0], round(s, 3), span in chunks[i][1]) for i, s in hits])

    chunks = fixed_chunks(30, 5)
    hits = retrieve(chunks, build_index(chunks), terms(q), 3)
    reranked = sorted(hits, key=lambda h: -rerank_score(chunks[h[0]][1], q))
    print("30-word chunks after rerank:", [(chunks[i][0], rerank_score(chunks[i][1], q)) for i, _ in reranked])

    miss = QUERIES[4][0]
    chunks = fixed_chunks(20, 5)
    idx = build_index(chunks)
    print(f"\n== The miss: {miss!r} ==")
    print("terms:", terms(miss), "-> hits:", retrieve(chunks, idx, terms(miss), 3))
    print("rewritten:", rewrite(miss))
    print("-> top hit:", [(chunks[i][0], round(s, 3), "not been accessed" in chunks[i][1]) for i, s in retrieve(chunks, idx, rewrite(miss), 1)])

    print("\n== Evidence recall@1 over the 5 questions ==")
    for name, ch in [("fixed 20w/5 overlap", fixed_chunks(20, 5)), ("sentences <= 20w", sentence_chunks(20))]:
        print(f"{name:20s}: {len(ch):2d} chunks, recall@1 {recall_at(ch, 1):.1f}, with rewrite {recall_at(ch, 1, True):.1f}")

    print("\n== Grounded prompt (20-word chunks, k = 2) ==")
    chunks = fixed_chunks(20, 5)
    hits = retrieve(chunks, build_index(chunks), terms(q), 2)
    print(build_prompt(q, chunks, hits))
    print("\ncitation check on 'Yes, if it arrived damaged [1][4].':", check_citations("Yes, if it arrived damaged [1][4].", len(hits)))
