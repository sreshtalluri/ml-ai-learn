# Adapted from AI Engineering from Scratch (MIT), phases/05-nlp-foundations-to-advanced/06-named-entity-recognition
"""Named entity recognition from scratch: BIO tags, word-shape features, a gazetteer baseline,
a per-token classifier, Viterbi decoding with BIO transition constraints, and entity-level F1.

All sentences are SYNTHETIC (templates filled with made-up and real-looking names).
Fixed seed, offline, runs in a few seconds.
Run from guide/:  uv run code/09-classical-nlp/ner_scratch.py
"""
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402


# ---------------------------------------------------------------------------
# 1. BIO <-> spans
# ---------------------------------------------------------------------------
def spans_to_bio(n_tokens, spans):
    labels = ["O"] * n_tokens
    for start, end, typ in spans:                 # end is exclusive
        labels[start] = f"B-{typ}"
        for i in range(start + 1, end):
            labels[i] = f"I-{typ}"
    return labels


def bio_to_spans(labels):
    spans, cur = [], None
    for i, lab in enumerate(labels + ["O"]):      # sentinel "O" closes a span at the end
        if cur and not (lab.startswith("I-") and lab[2:] == cur[2]):
            spans.append(tuple(cur))
            cur = None
        if lab.startswith("B-") or (lab.startswith("I-") and cur is None):  # lenient: stray I- starts a span
            cur = [i, i + 1, lab[2:]]
        elif cur:
            cur[1] = i + 1
    return spans


# ---------------------------------------------------------------------------
# 2. Features: word shape and context
# ---------------------------------------------------------------------------
def word_shape(w):
    return "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in w)


def token_features(tokens, i, lexical=False):
    """Shape and context features. lexical=True also adds the word itself (lets the model memorise names)."""
    w = tokens[i]
    ctx = lambda j: tokens[j].lower() if 0 <= j < len(tokens) else ("<BOS>" if j < 0 else "<EOS>")  # noqa: E731
    f = {"shape=" + word_shape(w): 1, "is_title": w.istitle(), "is_upper": w.isupper(), "first": i == 0,
         "prev=" + ctx(i - 1): 1, "next=" + ctx(i + 1): 1, "prev2=" + ctx(i - 2): 1, "next2=" + ctx(i + 2): 1,
         "prev_title": i > 0 and tokens[i - 1].istitle()}
    if lexical:
        f["word=" + w.lower()] = 1
        f["suffix3=" + w[-3:].lower()] = 1
    return f


# ---------------------------------------------------------------------------
# 3. SYNTHETIC corpus: templates + name lists (train and test names do not overlap)
# ---------------------------------------------------------------------------
NAMES = {
    "train": {"PER": ["Ada Lovelace", "Alan Turing", "Grace", "Linus", "Maria Silva", "Kenji Sato"],
              "ORG": ["Acme Corp", "Globex", "Initech", "Umbrella Labs", "Stark Industries"],
              "LOC": ["Paris", "New York", "Lagos", "Mumbai", "Buenos Aires", "Oslo"]},
    "test": {"PER": ["Rosa Parks", "Tim Berners Lee", "Fatima", "Omar Haddad"],
             "ORG": ["Hooli", "Wayne Enterprises", "Cyberdyne", "Vandelay Industries"],
             "LOC": ["Nairobi", "San Francisco", "Kyoto", "Rio de Janeiro"]},
}
TEMPLATES = [
    "{PER} joined {ORG} in {LOC} .", "{ORG} opened an office in {LOC} .", "yesterday {PER} visited {LOC} .",
    "{PER} said {ORG} will hire engineers .", "the mayor of {LOC} met {PER} .", "{ORG} acquired {ORG} last year .",
    "shares of {ORG} fell after {PER} resigned .", "{PER} moved from {LOC} to {LOC} .",
]


def make_corpus(split, n, rng):
    data = []
    for _ in range(n):
        tokens, spans = [], []
        for piece in rng.choice(TEMPLATES).split():
            if piece.startswith("{"):
                typ = piece[1:-1]
                name = rng.choice(NAMES[split][typ]).split()
                spans.append((len(tokens), len(tokens) + len(name), typ))
                tokens += name
            else:
                tokens.append(piece)
        if tokens[0][0].islower():
            tokens[0] = tokens[0].capitalize()
        data.append((tokens, spans_to_bio(len(tokens), spans)))
    return data


# ---------------------------------------------------------------------------
# 4. Viterbi with BIO constraints (log-space)
# ---------------------------------------------------------------------------
def allowed(prev, cur):
    """I-X may only follow B-X or I-X. Everything else is allowed."""
    return not cur.startswith("I-") or (prev != "O" and prev[2:] == cur[2:])


def viterbi(log_emit, tags):
    n, k = log_emit.shape
    trans = np.array([[0.0 if allowed(p, c) else -np.inf for c in tags] for p in tags])   # [k, k]
    start = np.array([-np.inf if t.startswith("I-") else 0.0 for t in tags])
    score = start + log_emit[0]                                   # [k] best score ending in each tag
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        cand = score[:, None] + trans + log_emit[i][None, :]       # [k_prev, k_cur]
        back[i] = cand.argmax(0)
        score = cand.max(0)
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return [tags[j] for j in reversed(path)]


# ---------------------------------------------------------------------------
# 5. Metrics: token accuracy vs entity-level P/R/F1 (exact span + type match)
# ---------------------------------------------------------------------------
def entity_prf(gold_seqs, pred_seqs):
    tp = n_pred = n_gold = 0
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = set(bio_to_spans(g)), set(bio_to_spans(p))
        tp += len(gs & ps)
        n_pred += len(ps)
        n_gold += len(gs)
    prec, rec = tp / max(n_pred, 1), tp / max(n_gold, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)


def token_acc(gold_seqs, pred_seqs):
    pairs = [(a, b) for g, p in zip(gold_seqs, pred_seqs) for a, b in zip(g, p)]
    return sum(a == b for a, b in pairs) / len(pairs)


def invalid_count(seqs):
    return sum(not allowed(p, c) for s in seqs for p, c in zip(["O"] + s[:-1], s))


def main():
    print("== BIO round trip ==")
    toks = "The New York City mayor visited Initech .".split()
    bio = spans_to_bio(len(toks), [(1, 4, "LOC"), (6, 7, "ORG")])
    print(list(zip(toks, bio)))
    print("spans:", bio_to_spans(bio))
    print("shapes:", {w: word_shape(w) for w in ["Apple", "iPhone", "IBM", "USA-2024", "apple"]})

    print("\n== Worked example: greedy vs Viterbi on 'New York' ==")
    tags = ["O", "B-LOC", "I-LOC"]
    probs = np.array([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5]])          # rows: New, York
    print("greedy:", [tags[j] for j in probs.argmax(1)], "(I-LOC after O is invalid)")
    with np.errstate(divide="ignore"):
        print("viterbi:", viterbi(np.log(probs), tags), f"score {0.4 * 0.5:.2f} vs best valid alternative {0.6 * 0.3:.2f}")

    print("\n== Worked example: token accuracy vs entity F1 ==")
    gold = ["B-LOC", "I-LOC", "I-LOC", "O", "B-ORG"]
    pred = ["B-LOC", "I-LOC", "O", "O", "B-ORG"]
    print(f"token accuracy {token_acc([gold], [pred]):.2f}, entity P/R/F1 = "
          + "/".join(f"{v:.2f}" for v in entity_prf([gold], [pred])))

    rng = random.Random(0)
    train, test = make_corpus("train", 400, rng), make_corpus("test", 200, rng)
    test_gold = [lab for _, lab in test]

    # Gazetteer baseline: exact lookup of the TRAINING name lists
    gaz = {tuple(n.split()): t for t, names in NAMES["train"].items() for n in names}

    def gazetteer(tokens):
        labels, i = ["O"] * len(tokens), 0
        while i < len(tokens):
            for L in (3, 2, 1):
                if tuple(tokens[i:i + L]) in gaz:
                    labels[i:i + L] = spans_to_bio(L, [(0, L, gaz[tuple(tokens[i:i + L])])])
                    i += L - 1
                    break
            i += 1
        return labels

    # Trained per-token classifiers on hand features: with and without the word identity
    def fit(lexical):
        dv = DictVectorizer()
        X = dv.fit_transform([token_features(t, i, lexical) for t, _ in train for i in range(len(t))])
        clf = LogisticRegression(max_iter=2000).fit(X, [lab for _, labs in train for lab in labs])
        return dv, clf, [str(c) for c in clf.classes_]

    def predict(model, lexical, use_viterbi):
        dv, clf, tag_list = model
        out = []
        for tokens, _ in test:
            P = clf.predict_proba(dv.transform([token_features(tokens, i, lexical) for i in range(len(tokens))]))
            out.append(viterbi(np.log(P + 1e-12), tag_list) if use_viterbi else [tag_list[j] for j in P.argmax(1)])
        return out

    lex_model, ctx_model = fit(lexical=True), fit(lexical=False)
    print("\n== SYNTHETIC test set: 200 sentences, names never seen in training ==")
    print(f"{'model':>30} | {'token acc':>9} | {'entity P':>8} | {'entity R':>8} | {'entity F1':>9} | {'invalid':>7}")
    rows = {}
    for name, pred in [("gazetteer (training names)", [gazetteer(t) for t, _ in test]),
                       ("word+context, Viterbi", predict(lex_model, True, True)),
                       ("shape+context, greedy", predict(ctx_model, False, False)),
                       ("shape+context, Viterbi", predict(ctx_model, False, True))]:
        p, r, f = entity_prf(test_gold, pred)
        rows[name] = (token_acc(test_gold, pred), f, pred)
        print(f"{name:>30} | {rows[name][0]:>9.3f} | {p:>8.3f} | {r:>8.3f} | {f:>9.3f} | {invalid_count(pred):>7}")
    vit_pred = rows["shape+context, Viterbi"][2]

    print("\nexample:")
    tokens = test[3][0]
    print("tokens :", tokens)
    print("gold   :", test_gold[3])
    print("viterbi:", vit_pred[3])

    # ---------------------------- figure ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    names = list(rows)
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [rows[n][0] for n in names], 0.4, color=GRAY, label="token accuracy")
    ax[0].bar(x + 0.2, [rows[n][1] for n in names], 0.4, color=BLUE, label="entity F1")
    ax[0].set_xticks(x, ["gazetteer", "word+ctx\nViterbi", "shape+ctx\ngreedy", "shape+ctx\nViterbi"])
    ax[0].set(ylim=(0, 1.05), title="Unseen names (SYNTHETIC test set)")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[1].axis("off")
    ax[1].set_title("BIO tags for one sentence")
    colors = {"PER": ORANGE, "ORG": TEAL, "LOC": BLUE}
    for i, (w, t) in enumerate(zip(toks, bio)):
        c = colors.get(t[2:], GRAY) if t != "O" else GRAY
        ax[1].text(0.02, 0.9 - i * 0.11, w, fontsize=10, transform=ax[1].transAxes)
        ax[1].text(0.55, 0.9 - i * 0.11, t, fontsize=10, color=c, weight="bold", transform=ax[1].transAxes)
    save(fig, "named-entity-recognition")


if __name__ == "__main__":
    main()
