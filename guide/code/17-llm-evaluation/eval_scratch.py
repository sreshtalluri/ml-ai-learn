# Adapted from AI Engineering from Scratch (MIT), phases/19-capstone-projects/68-rag-eval-precision-recall, phases/19-capstone-projects/72-code-exec-metric, phases/19-capstone-projects/73-perplexity-calibration, phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks, phases/11-llm-engineering/10-evaluation
"""LLM evaluation metrics from scratch: ranking metrics, exact match and token F1, a claim-level
faithfulness check, pass@k for code, expected calibration error, and a paired bootstrap.

Synthetic: every case, label, and confidence below is made up to show the mechanics.
Run from guide/:  uv run code/17-llm-evaluation/eval_scratch.py
"""
import math
import re
import string
from collections import Counter

import numpy as np


# --- 1. Ranking metrics -------------------------------------------------------------
def recall_at_k(rels, k, R):
    return sum(rels[:k]) / R


def reciprocal_rank(rels):
    return next((1 / (i + 1) for i, r in enumerate(rels) if r), 0.0)   # 0 if nothing relevant


def ndcg_at_k(rels, k):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))   # rank i+1 -> log2(i+2)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0


# --- 2. Exact match and token F1 (SQuAD-style normalization) -------------------------
def normalize(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in s.split() if t not in {"a", "an", "the"}]


def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    p, r = normalize(pred), normalize(ref)
    overlap = sum((Counter(p) & Counter(r)).values())   # shared tokens, counted with multiplicity
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


# --- 3. Faithfulness: fraction of answer claims the context supports -----------------
def faithfulness(answer, context, supported):
    """`supported(claim, context) -> bool` is an NLI model or LLM judge in production."""
    claims = [c.strip() for c in re.split(r"(?<=[.!?])\s+", answer) if c.strip()]
    verdicts = [supported(c, context) for c in claims]
    return sum(verdicts) / len(claims), list(zip(claims, verdicts))


def overlap_judge(claim, context, threshold=0.6):   # a crude stand-in for an NLI model
    words = [w for w in normalize(claim) if len(w) > 2]
    return sum(w in normalize(context) for w in words) / max(1, len(words)) >= threshold


# --- 4. pass@k for generated code ------------------------------------------------------
def pass_at_k(n, c, k):
    """Unbiased estimate: chance at least one of k samples passes, from n samples with c passing."""
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


# --- 5. Expected calibration error ----------------------------------------------------
def ece(conf, correct, bins=5):
    conf, correct = np.asarray(conf), np.asarray(correct)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():                               # weight each bin's |accuracy - confidence| by its size
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


# --- 6. Paired bootstrap for a version comparison --------------------------------------
def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), (n_boot, len(a)))   # resample CASES; both versions share them
    return np.percentile(b[idx].mean(1) - a[idx].mean(1), [2.5, 97.5])


if __name__ == "__main__":
    print("== Worked example 1: relevance [0, 1, 0, 1, 0], R = 2 ==")
    rels = [0, 1, 0, 1, 0]
    print("recall@1,3,5:", [recall_at_k(rels, k, 2) for k in (1, 3, 5)],
          f"RR {reciprocal_rank(rels):.3f}  nDCG@5 {ndcg_at_k(rels, 5):.3f}")
    rrs = [reciprocal_rank(r) for r in ([1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1])]
    print(f"MRR over 4 queries (first relevant at 1, 2, none, 3): {np.mean(rrs):.3f}")

    print("\n== Worked example 2: answer vs reference ==")
    pred, ref = "within 30 days", "30 days of purchase"
    print(f"EM {exact_match(pred, ref):.0f}  F1 {token_f1(pred, ref):.3f}  |  "
          f"EM('The 30 days of purchase.') {exact_match('The 30 days of purchase.', ref):.0f}")

    print("\n== Faithfulness (synthetic RAG answer) ==")
    context = "Refunds are issued to the original payment method within 5 business days."
    answer = "Refunds go to the original payment method. They arrive within 5 business days. A store credit option is also available."
    score, detail = faithfulness(answer, context, overlap_judge)
    for claim, ok in detail:
        print(f"  {'supported  ' if ok else 'UNSUPPORTED'} | {claim}")
    print(f"faithfulness = {score:.2f}")

    print("\n== pass@k: 20 samples per problem, 3 pass ==")
    print("  ".join(f"pass@{k} = {pass_at_k(20, 3, k):.3f}" for k in (1, 5, 10)))

    print("\n== Calibration: 10 answers with stated confidence (synthetic) ==")
    conf = [0.95, 0.9, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    correct = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    print(f"mean confidence {np.mean(conf):.2f}, accuracy {np.mean(correct):.2f}, ECE {ece(conf, correct):.3f}")

    print("\n== Worked example 3 shape: 40 cases, v2 fixes 6 and breaks 2 ==")
    v1 = np.array([1] * 30 + [0] * 10)
    v2 = v1.copy(); v2[30:36] = 1; v2[:2] = 0       # 6 fixed, 2 broken -> 0.75 to 0.85
    lo, hi = paired_bootstrap(v1, v2)
    print(f"accuracy {v1.mean():.2f} -> {v2.mean():.2f}; paired 95% CI for the gain [{lo:+.3f}, {hi:+.3f}]")
