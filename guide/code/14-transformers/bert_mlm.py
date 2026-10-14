# Adapted from AI Engineering from Scratch (MIT), phases/07-transformers-deep-dive/06-bert-masked-language-modeling
"""BERT-style masked language modeling: the 80/10/10 masking rule, whole-word masking,
the loss mask, and why seeing both sides of a blank helps (count-based predictor on a synthetic corpus).

Run from guide/:  uv run code/14-transformers/bert_mlm.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

VOCAB = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "the", "cat", "dog", "sat", "ran", "on", "in",
         "mat", "park", "a", "big", "small", "red", "fast", "home", "went"]
ID = {w: i for i, w in enumerate(VOCAB)}
SPECIAL = {ID["[PAD]"], ID["[CLS]"], ID["[SEP]"], ID["[MASK]"]}
IGNORE = -100                                      # positions the loss skips


# ---------------------------------------------------------------------------
# 1. The 80/10/10 rule
# ---------------------------------------------------------------------------
def mask_tokens(ids, rng, p=0.15):
    ids = np.array(ids)
    labels = np.full_like(ids, IGNORE)
    can = ~np.isin(ids, list(SPECIAL))             # never mask [CLS], [SEP], [PAD]
    chosen = can & (rng.random(len(ids)) < p)      # pick ~15% of positions
    labels[chosen] = ids[chosen]                   # loss only on chosen positions
    r = rng.random(len(ids))
    inputs = ids.copy()
    inputs[chosen & (r < 0.8)] = ID["[MASK]"]      # 80%: [MASK]
    rand = chosen & (r >= 0.8) & (r < 0.9)         # 10%: a random real token
    inputs[rand] = rng.integers(len(SPECIAL), len(VOCAB), rand.sum())
    return inputs, labels                          # remaining 10%: unchanged


rng = np.random.default_rng(0)
sent = ["[CLS]", "the", "small", "cat", "sat", "on", "the", "red", "mat", "[SEP]"]
inp, lab = mask_tokens([ID[w] for w in sent], rng, p=0.4)
print("=== one sentence, p=0.4 so the effect is visible ===")
for w, i, l in zip(sent, inp, lab):
    print(f"  {w:>6} -> input {VOCAB[i]:>7}   label {VOCAB[l] if l != IGNORE else '-100 (ignored)'}")

# Check the proportions on 200,000 synthetic token ids.
ids = rng.integers(len(SPECIAL), len(VOCAB), 200_000)
inp, lab = mask_tokens(ids, rng)
sel = lab != IGNORE
masked = (inp[sel] == ID["[MASK]"]).mean()
same = (inp[sel] == ids[sel]).mean()
print(f"\nselected {sel.mean():.2%} of tokens; of those: [MASK] {masked:.2%}, "
      f"random {1 - masked - same:.2%}, unchanged {same:.2%}")
print("(a random draw can pick the original token, so 'unchanged' sits slightly above 10%)")


# ---------------------------------------------------------------------------
# 2. Whole-word masking: mask every subword of a chosen word together
# ---------------------------------------------------------------------------
def whole_word_spans(pieces):
    """Group WordPiece-style tokens: a piece starting with '##' continues the previous word."""
    spans, start = [], 0
    for k in range(1, len(pieces) + 1):
        if k == len(pieces) or not pieces[k].startswith("##"):
            spans.append((start, k))
            start = k
    return spans


pieces = ["the", "un", "##believ", "##able", "cat", "play", "##ed"]
print("\nwhole-word spans:", [" ".join(pieces[a:b]) for a, b in whole_word_spans(pieces)])


# ---------------------------------------------------------------------------
# 3. Why bidirectional: fill a blank from the left only, or from both sides
# ---------------------------------------------------------------------------
subjects, verbs, places = ["cat", "dog"], ["sat", "ran", "went"], ["mat", "park", "home"]
VERB_PLACE = {"sat": ["mat"], "ran": ["park"], "went": ["home"]}   # synthetic grammar
gen = np.random.default_rng(1)


def make_sentence():
    v = gen.choice(verbs)
    prep = "on" if v == "sat" else "in" if v == "ran" else "the"
    return ["the", gen.choice(subjects), v, prep, "the", VERB_PLACE[v][0]] if prep != "the" \
        else ["the", gen.choice(subjects), v, "home"]


train = [make_sentence() for _ in range(2000)]
test = [make_sentence() for _ in range(500)]
left, both = Counter(), Counter()
for s in train:
    for k in range(1, len(s)):
        left[(s[k - 1], s[k])] += 1                       # left neighbour only (GPT-style)
        right = s[k + 1] if k + 1 < len(s) else "</s>"
        both[(s[k - 1], right, s[k])] += 1                 # left AND right (BERT-style)


def guess_left(prev):
    return max((c, w) for (p, w), c in left.items() if p == prev)[1]


def guess_both(prev, nxt):
    return max((c, w) for (p, r, w), c in both.items() if p == prev and r == nxt)[1]


hits_l = hits_b = 0
for s in test:                                          # blank out the verb, position 2
    hits_l += guess_left(s[1]) == s[2]
    hits_b += guess_both(s[1], s[3]) == s[2]
acc_l, acc_b = hits_l / len(test), hits_b / len(test)
print(f"\nfill the verb blank on {len(test)} synthetic test sentences:")
print(f"  left context only : {acc_l:.1%}")
print(f"  both sides        : {acc_b:.1%}")


# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
a1.bar(["[MASK]", "random token", "unchanged"], [masked, 1 - masked - same, same],
       color=[BLUE, ORANGE, TEAL])
a1.set_title("What happens to the 15% selected tokens\n(200,000 synthetic tokens)")
a1.set_ylabel("share of selected tokens"); a1.set_ylim(0, 1)
for x, v in enumerate([masked, 1 - masked - same, same]):
    a1.text(x, v + 0.02, f"{v:.1%}", ha="center")
a2.bar(["left context only\n(causal)", "both sides\n(bidirectional)"], [acc_l, acc_b], color=[GRAY, BLUE])
a2.set_title("Filling a blank verb in synthetic sentences")
a2.set_ylabel("accuracy"); a2.set_ylim(0, 1.1)
for x, v in enumerate([acc_l, acc_b]):
    a2.text(x, v + 0.02, f"{v:.1%}", ha="center")
fig.tight_layout()
save(fig, "bert-masked-language-modeling")
