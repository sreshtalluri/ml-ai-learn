---
title: Named entity recognition
summary: Turn entity extraction into per-token BIO labels, decode valid tag sequences with Viterbi, and evaluate with entity-level F1 instead of token accuracy.
skill: nlp
minutes: 40
prerequisites: [text-to-vectors, word-embeddings]
related: [logistic-regression, recurrent-networks, bert-masked-language-modeling, classification-metrics]
adapted_from: [phases/05-nlp-foundations-to-advanced/06-named-entity-recognition]
---

# Named entity recognition

> **Mental model.** Named entity recognition (NER) is highlighting with colored markers. You read a sentence and mark "Ada Lovelace" as a person, "Acme Corp" as an organization, and "New York" as a place. A model does it one token at a time, with a label that says "a new highlight starts here", "the highlight continues", or "no highlight". A decoder then makes sure no highlight continues one that never started.

**You will learn to**
- Convert labeled spans to BIO tags and back, and explain why BIO turns span extraction into token classification.
- Build hand-crafted token features (word shape, context words) and say which ones generalize to unseen names.
- Decode the best valid tag sequence with the Viterbi algorithm, and show why greedy per-token decoding produces invalid sequences.
- Compute entity-level precision, recall, and F1, and explain why token accuracy overstates quality.
- Choose between a gazetteer, a CRF, a fine-tuned transformer, and an LLM prompt for a given extraction job.

**Why it matters.** NER sits underneath resume parsing, PII redaction, search query understanding, contract and clinical-note extraction, and knowledge-graph building. It is also the cleanest example of *sequence labeling*: the label of one token depends on the labels around it. The same ideas (BIO tags, constrained decoding, span-level metrics) come back in slot filling, chunking, and structured extraction with LLMs.

## 1. Intuition

**Spans become tags.** An entity is a span of tokens with a type: `(1, 4, LOC)` means tokens 1, 2, and 3 form a location. Classifiers predict one label per input, so we give every token a label:

- `B-TYPE`: this token **b**egins an entity of TYPE.
- `I-TYPE`: this token is **i**nside (continues) an entity of TYPE.
- `O`: this token is **o**utside every entity.

"The New York City mayor visited Initech ." becomes `O B-LOC I-LOC I-LOC O O B-ORG O`. Two entities next to each other stay separate because the second one starts with a fresh `B-`.

**What a token looks like matters more than what it is.** A model that memorizes "Paris is a location" fails on "Nairobi". Features that generalize describe the token's *shape* (capitalized? all caps? digits?) and its *context* ("the mayor of ___", "___ joined", "shares of ___").

**Labels depend on neighbors.** `I-LOC` only makes sense after `B-LOC` or `I-LOC`. A per-token classifier doesn't know that, so it can output `O I-LOC`. A **decoder** picks the best whole sequence that obeys the rules. Linear-chain CRFs learn these transition scores; we use hard rules plus the Viterbi algorithm.

**Score entities, not tokens.** Most tokens are `O`, and getting `O` right is easy. If the model finds "New York" but misses "City", 2 of 3 tokens are right, but the entity is wrong: a downstream system gets the wrong place. NER is scored on exact span-and-type matches.

## 2. Visualization

![Left: on a synthetic test set whose names never appear in training, a gazetteer gets 0.52 token accuracy and 0.00 entity F1; a classifier with word-identity features gets 0.82 and 0.56; a shape-and-context classifier with greedy decoding gets 0.89 and 0.66; the same classifier with Viterbi decoding gets 0.91 and 0.73. Right: the sentence "The New York City mayor visited Initech ." with BIO tags O B-LOC I-LOC I-LOC O O B-ORG O.](../../figures/named-entity-recognition.png)

*Synthetic data: 400 training and 200 test sentences built from templates, with disjoint name lists, so every test entity is unseen. Token accuracy (gray) is always far above entity F1 (blue). The gazetteer's 0.52 token accuracy comes from predicting `O` everywhere: about half the tokens are `O`.*

**Try it** (predict first, then run the script and check):
1. The gazetteer scores 0 entity F1. Predict its F1 if the test set reused the training names. Change `make_corpus("test", ...)` to use `"train"` and rerun.
2. Adding the word itself as a feature *lowers* entity F1 (0.73 to 0.56 with Viterbi). Predict why before reading section 5.
3. Greedy decoding produces 38 invalid transitions. Predict how many Viterbi produces, and why it must be that number.
4. Add a template like "{PER} , CEO of {ORG} , said ..." to the test set only. Predict which entity type suffers most.

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $x = (x_1, \dots, x_n)$ | tokens of one sentence | $n$ |
| $y = (y_1, \dots, y_n)$ | one tag per token, from a tag set of size $K$ | $n$ |
| $e_i(k)$ | emission score: log-probability the classifier gives tag $k$ at position $i$ | $[n, K]$ |
| $A(j, k)$ | transition score from tag $j$ to tag $k$; $-\infty$ for forbidden pairs | $[K, K]$ |
| $\delta_i(k)$ | best total score of any tag sequence for $x_1..x_i$ that ends in tag $k$ | $[n, K]$ |
| $P, R, F_1$ | entity-level precision, recall, F1 | scalars |

### Sequence score and the BIO constraint

A tag sequence scores

```math
s(y) = \sum_{i=1}^{n} e_i(y_i) + \sum_{i=2}^{n} A(y_{i-1}, y_i), \qquad A(j, k) = \begin{cases} -\infty & k = \text{I-}X \text{ and } j \notin \{\text{B-}X, \text{I-}X\} \\ 0 & \text{otherwise} \end{cases}
```

A sentence also may not start with `I-X`. A linear-chain CRF uses the same score but *learns* $A$ and the emission weights jointly, normalizing over all $K^n$ sequences: $p(y \mid x) = \exp s(y) / \sum_{y'} \exp s(y')$.

### Viterbi

Trying all $K^n$ sequences is impossible ($K = 7$, $n = 20$ gives $8 \times 10^{16}$). Viterbi keeps only the best score for each ending tag:

```math
\delta_1(k) = e_1(k), \qquad \delta_i(k) = e_i(k) + \max_j \big[\delta_{i-1}(j) + A(j, k)\big]
```

It stores the winning $j$ for each $(i, k)$, picks $\arg\max_k \delta_n(k)$, and follows the pointers back. Cost: $O(n K^2)$.

### Worked example: greedy versus Viterbi on "New York"

Tags `O`, `B-LOC`, `I-LOC`. The classifier's probabilities:

| Token | O | B-LOC | I-LOC |
|---|---|---|---|
| New | 0.6 | 0.4 | 0.0 |
| York | 0.2 | 0.3 | 0.5 |

**Greedy** takes each row's maximum: `O`, `I-LOC`. That is invalid: `I-LOC` continues an entity that never began.

**Viterbi** compares valid sequences by the product of probabilities (a sum in log space):

- `O O`: $0.6 \times 0.2 = 0.12$
- `O B-LOC`: $0.6 \times 0.3 = 0.18$
- `B-LOC O`: $0.4 \times 0.2 = 0.08$
- `B-LOC B-LOC`: $0.4 \times 0.3 = 0.12$
- `B-LOC I-LOC`: $0.4 \times 0.5 = 0.20$
- `O I-LOC`: forbidden, score $-\infty$.

The best is `B-LOC I-LOC` at 0.20, which beats `O B-LOC` at 0.18. Viterbi found "New York" as one location even though `O` was the more likely tag for "New" on its own.

### Worked example: token accuracy versus entity F1

Gold `B-LOC I-LOC I-LOC O B-ORG` ("New York City ... Initech"); prediction `B-LOC I-LOC O O B-ORG`.

- Token accuracy: positions 1, 2, 4, 5 match, so $4/5 = 0.80$.
- Gold entities: $\{(0,3,\text{LOC}), (4,5,\text{ORG})\}$. Predicted: $\{(0,2,\text{LOC}), (4,5,\text{ORG})\}$. Exact matches: 1.
- $P = 1/2 = 0.50$, $R = 1/2 = 0.50$, $F_1 = 2 \times 0.5 \times 0.5 / (0.5 + 0.5) = 0.50$.

The truncated location counts as both a false positive (the wrong span was predicted) and a false negative (the right span was missed).

## 4. Implementation

The full script is [`code/09-classical-nlp/ner_scratch.py`](../../code/09-classical-nlp/ner_scratch.py) (NumPy plus scikit-learn's logistic regression as the token classifier).

### Build it from scratch

**Step 1: spans to BIO tags.** Training data usually arrives as character or token spans. We write the first token of each span as `B-` and the rest as `I-`. The end index is exclusive, like Python slices.

```python
def spans_to_bio(n_tokens, spans):
    labels = ["O"] * n_tokens
    for start, end, typ in spans:                 # end is exclusive
        labels[start] = f"B-{typ}"
        for i in range(start + 1, end):
            labels[i] = f"I-{typ}"
    return labels
```

*Predict:* two adjacent organizations, "Globex Initech", with spans `(0,1,ORG)` and `(1,2,ORG)`. What tags come out, and could you tell them apart with IO tags (no `B-`)?

**Step 2: BIO tags back to spans.** Predictions are tags, but users and metrics want spans. A span closes when the next tag is not an `I-` of the same type. The sentinel `"O"` closes a span that runs to the end. A stray `I-X` with no open span is treated as a new span; this is the lenient convention, and strict evaluators reject it instead.

```python
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
```

*Predict:* what does `bio_to_spans(["B-PER", "I-ORG"])` return?

**Step 3: features that generalize.** Word shape maps every uppercase letter to `X`, lowercase to `x`, and digits to `d`, so "iPhone" becomes `xXxxxx` and "USA-2024" becomes `XXX-dddd`. Context words two to the left and right capture patterns like "the mayor of ___". The `lexical` switch adds the word itself, which we will see hurts on unseen names.

```python
def word_shape(w):
    return "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in w)

def token_features(tokens, i, lexical=False):
    w = tokens[i]
    ctx = lambda j: tokens[j].lower() if 0 <= j < len(tokens) else ("<BOS>" if j < 0 else "<EOS>")
    f = {"shape=" + word_shape(w): 1, "is_title": w.istitle(), "is_upper": w.isupper(), "first": i == 0,
         "prev=" + ctx(i - 1): 1, "next=" + ctx(i + 1): 1, "prev2=" + ctx(i - 2): 1, "next2=" + ctx(i + 2): 1,
         "prev_title": i > 0 and tokens[i - 1].istitle()}
    if lexical:
        f["word=" + w.lower()] = 1
        f["suffix3=" + w[-3:].lower()] = 1
    return f
```

*Predict:* the first word of every sentence is capitalized. Which feature lets the model tell "Yesterday" (not an entity) from "Grace" (a person) at position 0?

**Step 4: Viterbi with BIO constraints.** The transition matrix holds 0 for allowed pairs and $-\infty$ for forbidden ones. `cand[j, k]` is the best score ending in tag `j` at the previous token plus the move to `k`. We keep the max over `j` and remember which `j` won.

```python
def allowed(prev, cur):
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
```

*Predict:* if you replace $-\infty$ with $-2$ in `trans`, can Viterbi still output an invalid sequence? When?

**Step 5: entity-level metrics.** Convert both sequences to sets of `(start, end, type)` and count exact matches. Precision divides by predicted entities, recall by gold entities.

```python
def entity_prf(gold_seqs, pred_seqs):
    tp = n_pred = n_gold = 0
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = set(bio_to_spans(g)), set(bio_to_spans(p))
        tp += len(gs & ps)
        n_pred += len(ps)
        n_gold += len(gs)
    prec, rec = tp / max(n_pred, 1), tp / max(n_gold, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)
```

*Predict:* a model that outputs `O` for every token. What are its precision, recall, and token accuracy on a test set where 48% of tokens are entities?

The script then trains `LogisticRegression` on the feature dictionaries (via `DictVectorizer`) and decodes its `predict_proba` output greedily and with Viterbi.

### Use the library

The classic trained baseline is a linear-chain CRF over the same feature dictionaries. `sklearn-crfsuite` learns emission and transition weights together:

```python
import sklearn_crfsuite   # pip install sklearn-crfsuite

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
                           all_possible_transitions=True)   # learn weights for unseen pairs too (often negative)
crf.fit([[token_features(t, i) for i in range(len(t))] for t, _ in train],
        [labels for _, labels in train])
```

For production, a pretrained pipeline is usually the starting point:

```python
import spacy                                      # python -m spacy download en_core_web_sm
doc = spacy.load("en_core_web_sm")("Apple sued Google over its iPhone search deal in the US.")
print([(e.text, e.label_) for e in doc.ents])     # spans with OntoNotes labels such as ORG, GPE, PRODUCT

from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")  # merges B-/I- word pieces
print(ner("Apple sued Google over its iPhone in the US."))  # entity_group in {PER, ORG, LOC, MISC}

from seqeval.metrics import classification_report  # entity-level P/R/F1 per type
from seqeval.scheme import IOB2
print(classification_report(gold_tags, pred_tags, mode="strict", scheme=IOB2))
```

What the libraries add: learned transition weights (CRF), subword tokenization with labels aligned to word pieces (transformers), span merging (`aggregation_strategy`), and per-type reports (seqeval). Small models make type mistakes on things like product names, so always check on your own data.

### Run it

```bash
uv run code/09-classical-nlp/ner_scratch.py
```

Key lines of output:

```text
greedy: ['O', 'I-LOC'] (I-LOC after O is invalid)
viterbi: ['B-LOC', 'I-LOC'] score 0.20 vs best valid alternative 0.18
token accuracy 0.80, entity P/R/F1 = 0.50/0.50/0.50

                         model | token acc | entity P | entity R | entity F1 | invalid
    gazetteer (training names) |     0.517 |    0.000 |    0.000 |     0.000 |       0
         word+context, Viterbi |     0.817 |    0.562 |    0.553 |     0.557 |       0
         shape+context, greedy |     0.889 |    0.633 |    0.690 |     0.660 |      38
        shape+context, Viterbi |     0.912 |    0.706 |    0.755 |     0.730 |       0
```

The first three lines reproduce the worked examples. The table is on synthetic sentences whose names never appear in training.

## 5. Engineering

**The approach ladder.**

| Approach | Needs | Strengths | Weaknesses |
|---|---|---|---|
| Gazetteer and regex | name lists, patterns | precise on known names, IDs, emails; no training | zero recall on new names; can't disambiguate "Apple" |
| Feature classifier or CRF | a few thousand labeled sentences | fast on CPU, interpretable, stable | feature engineering; weaker on long-range context |
| Fine-tuned encoder (BERT-style) | hundreds to thousands of labeled sentences | strong accuracy, handles context | GPU for training, subword alignment |
| LLM prompting (zero/few-shot) | a label schema and a few examples | no training data; flexible label sets | latency and cost per call; inconsistent spans; needs validation |

A sound 2026 workflow: try an LLM prompt with a JSON schema to bootstrap labels, have humans correct a few hundred, measure entity F1 on a held-out set, then decide whether a small fine-tuned encoder meets the latency and cost budget. Regex still wins for structured IDs (emails, phone numbers, account numbers), and many PII pipelines combine regex with a model.

**Why the word feature hurt.** With word identity available, the classifier puts a lot of weight on "lovelace means PER". On unseen names those weights are useless, and the shape and context features got less weight during training. That is overfitting to the name list. In real data you want word features (they help on frequent entities), but you must evaluate on entities not seen in training, or your score measures memorization.

**Subword alignment.** Transformers split "Lovelace" into word pieces. Label only the first piece (and mask the rest with -100 in the loss), or copy the label to all pieces; then merge back to words at inference.

**Nested and overlapping entities.** "Bank of America Stadium" is a facility containing an organization. BIO assigns one tag per token, so it cannot represent both. Use span-based models (classify every candidate span) or one pass per type.

**Production concerns.** Domain shift is the main failure: a model trained on news misses drug names and contract parties. Re-label a sample from your domain before trusting any number. Track per-type F1, since rare types hide behind a good average. For PII redaction, optimize recall and accept extra false positives.

### Common mistakes

- Reporting token accuracy or token-level F1. Report entity-level P/R/F1 with exact span match, per type.
- Testing on sentences that share entity names with training data, which measures memorization.
- Greedy decoding that emits `I-X` after `O`, then a lenient span parser that silently "repairs" it.
- Treating LLM output as ground truth: offsets drift, spans get paraphrased. Validate that each extracted string actually occurs in the input.
- Forgetting that subword tokenizers break word-level labels.

## 6. Knowledge check

<!-- quiz:named-entity-recognition -->
**[Take the named entity recognition quiz](../../quizzes/named-entity-recognition.md)**
<!-- /quiz -->

**Practice exercise.** Tags `O`, `B-PER`, `I-PER`. For "Grace Hopper", the classifier gives "Grace": O 0.5, B-PER 0.45, I-PER 0.05; "Hopper": O 0.3, B-PER 0.1, I-PER 0.6. What do greedy decoding and Viterbi output?

<details>
<summary>Solution</summary>

Greedy: `O` (0.5) then `I-PER` (0.6), which is invalid. Viterbi scores the valid sequences: `O O` $= 0.5 \times 0.3 = 0.15$; `O B-PER` $= 0.5 \times 0.1 = 0.05$; `B-PER O` $= 0.45 \times 0.3 = 0.135$; `B-PER B-PER` $= 0.45 \times 0.1 = 0.045$; `B-PER I-PER` $= 0.45 \times 0.6 = 0.27$. Viterbi outputs `B-PER I-PER` (0.27), one person entity. (`I-PER` at the start is forbidden, so the 0.05 option never enters.)

</details>

**Implementation challenge.** Add BILOU tags (`L-` for the last token, `U-` for a single-token entity) to the script: write `spans_to_bilou`, update `allowed` for the new rules (for example, `I-X` and `L-X` must follow `B-X` or `I-X`), retrain, and compare entity F1 with BIO.

<details>
<summary>Hint</summary>

BILOU doubles the number of entity tags, so each tag has fewer training examples, but boundaries become explicit. Allowed transitions: `B-X` and `I-X` must be followed by `I-X` or `L-X`; `L-X`, `U-X`, and `O` must be followed by `O`, `B-*`, or `U-*`. The final token can't be `B-X` or `I-X`. You need an end-transition check as well as a start check.

</details>

## Summary

- BIO tags turn span extraction into per-token classification: `B-` starts, `I-` continues, `O` is outside.
- Shape and context features generalize to unseen names; word identity alone memorizes the training list.
- Per-token classifiers can emit invalid sequences; Viterbi finds the best valid sequence in $O(nK^2)$, and a CRF learns the transition scores.
- Evaluate with entity-level P/R/F1 (exact span and type) on entities unseen in training. Token accuracy is inflated by `O`.
- Pick the tool by data and budget: regex for IDs, CRF or a small encoder for fast and stable, LLM prompts to bootstrap.

**Next:** [Topic modeling](04-topic-modeling.md)

**Related:** [Word embeddings](02-word-embeddings.md) · [Logistic regression](../04-classification/01-logistic-regression.md) · [Recurrent networks](../13-deep-architectures/02-recurrent-networks.md) · [BERT and masked language modeling](../14-transformers/03-bert-masked-language-modeling.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Named Entity Recognition](https://aiengineeringfromscratch.com/lesson?path=phases/05-nlp-foundations-to-advanced/06-named-entity-recognition).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>How does BIO tagging turn entity extraction into a classification problem, and what can't it represent?</strong></summary>

An entity is a span with a type, but classifiers predict one label per input. BIO gives every token a label: B-TYPE for the first token of an entity, I-TYPE for each following token, and O for tokens outside any entity. "New York City" becomes B-LOC, I-LOC, I-LOC. Two adjacent entities of the same type stay separate because the second starts with a fresh B. Any per-token model can now be trained with cross-entropy, and spans are recovered by reading runs of B followed by I's. What BIO can't do is overlap: one tag per token means "Bank of America Stadium" can't be both a facility and contain an organization. Nested entities need span-based models that score every candidate span, or one pass per type. BILOU adds explicit last and unit tags, which sometimes helps boundary accuracy.

</details>

<details>
<summary><strong>Why use Viterbi or a CRF on top of a token classifier instead of taking the argmax at each position?</strong></summary>

Argmax decodes each token independently, so nothing stops it from emitting I-LOC after O, or I-ORG after B-PER. In the course script, greedy decoding of a logistic-regression tagger produced 38 invalid transitions on 200 test sentences. Viterbi finds the highest-scoring whole sequence under a transition matrix, with minus infinity for forbidden moves, in O(n K²) time by keeping the best score for each ending tag. It can overrule a locally likely tag: for "New York" with P(O) 0.6 and P(B-LOC) 0.4 on "New", but P(I-LOC) 0.5 on "York", the sequence B-LOC I-LOC scores 0.20 and beats every valid alternative. Here that raised entity F1 from 0.66 to 0.73. A linear-chain CRF goes further and learns soft transition weights jointly with the emissions, normalizing over all sequences.

</details>

<details>
<summary><strong>Your NER model reports 96% token accuracy but users say it misses half the entities. What's going on?</strong></summary>

Token accuracy is dominated by O tokens, which are usually the large majority and easy to get right. Predicting O everywhere can score very high when entities are sparse. Measure entity-level precision, recall, and F1 with exact span and type match, per entity type, using seqeval in strict mode. Then check where it fails. Boundary errors: "New York" found but "City" dropped counts as a miss, even though 2 of 3 tokens are correct. Unseen entities: if your test set shares names with training, you measured memorization; re-split so test entities are new. Domain shift: a news-trained model on contracts or clinical notes. Rare types hidden behind a good average. Also check the tokenizer: if subword labels are merged incorrectly, spans break. Fix the evaluation first, then label a domain sample.

</details>

<details>
<summary><strong>Design PII redaction for 50 million support tickets a day. Regex, CRF, fine-tuned transformer, or an LLM?</strong></summary>

50 million a day is about 580 tickets per second on average, more at peak, so per-ticket LLM calls are expensive and slow, and sending raw PII to an external API may itself violate policy. I'd layer it. Regex and checksums for structured PII (emails, phone numbers, card numbers with a Luhn check, account IDs): precise and nearly free. A small fine-tuned encoder (DistilBERT-size) for names and addresses, run on CPU or a modest GPU fleet in batches. An LLM is useful offline to pre-label training data, which humans then correct. Because a missed PII span is worse than an over-redaction, tune thresholds for high recall and report entity-level recall per type on a held-out set with unseen names. Monitor drift with periodic human audits of samples, and log only redacted text.

</details>
