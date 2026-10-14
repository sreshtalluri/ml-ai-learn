<!-- GENERATED from named-entity-recognition.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Named entity recognition

Covers the lesson [Named entity recognition](../lessons/09-classical-nlp/03-named-entity-recognition.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/named-entity-recognition/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What is BIO tagging?

- **A.** A tree representation of entities
- **B.** A binary entity versus non-entity label per token
- **C.** Per-token labels B-TYPE for the first token of an entity, I-TYPE for following tokens, O for outside
- **D.** A tokenization scheme for subwords

<details>
<summary>Answer</summary>

**C.** Per-token labels B-TYPE for the first token of an entity, I-TYPE for following tokens, O for outside

BIO turns span extraction into token classification. B starts an entity, I continues it, O is outside any entity.

- **A:** BIO is a flat sequence of labels, one per token, not a tree.
- **B:** A binary label loses both the entity type and where one entity ends and the next begins.
- **D:** BIO labels tokens after tokenization; it does not split text.

</details>

## 2. Multiple choice (easy)

Why are gazetteer (name-list) lookups brittle as the only NER method?

- **A.** They are slow to run
- **B.** They cannot match multi-token names
- **C.** They have no coverage of names missing from the list and cannot use context to disambiguate (Apple the company versus the fruit)
- **D.** They require a GPU

<details>
<summary>Answer</summary>

**C.** They have no coverage of names missing from the list and cannot use context to disambiguate (Apple the company versus the fruit)

Lookups only match known strings. In the course script a gazetteer of training names scored 0 entity F1 on a test set of unseen names.

- **A:** Hash lookups are very fast; speed is their strength.
- **B:** Matching the longest known token sequence first handles multi-token names.
- **D:** A lookup table runs anywhere.

</details>

## 3. Calculation (medium)

Tags O, B-LOC, I-LOC. Token "New": P(O) = 0.6, P(B-LOC) = 0.4, P(I-LOC) = 0. Token "York": P(O) = 0.2, P(B-LOC) = 0.3, P(I-LOC) = 0.5. I-LOC may only follow B-LOC or I-LOC. What is the probability product of the best **valid** two-tag sequence?

<details>
<summary>Answer</summary>

**0.2** (within ±0.001)

Valid sequences: O O = 0.12, O B-LOC = 0.18, B-LOC O = 0.08, B-LOC B-LOC = 0.12, B-LOC I-LOC = 0.4 × 0.5 = 0.20. O I-LOC (0.30) is forbidden. Best is B-LOC I-LOC at 0.20.

</details>

## 4. Calculation (medium)

Gold tags: B-LOC I-LOC I-LOC O B-ORG. Predicted: B-LOC I-LOC O O B-ORG. What is the entity-level F1 (exact span and type match)?

<details>
<summary>Answer</summary>

**0.5** (within ±0.001)

Gold spans {(0,3,LOC), (4,5,ORG)}; predicted {(0,2,LOC), (4,5,ORG)}. One exact match. P = 1/2, R = 1/2, F1 = 0.5. Token accuracy is 4/5 = 0.8, which overstates quality.

</details>

## 5. Multiple choice (medium)

In a BiLSTM-CRF or a feature-based linear-chain CRF, what does the CRF part add over an independent per-token classifier?

- **A.** It tokenizes the text
- **B.** It learns transition scores between tags and decodes the best whole sequence, so invalid sequences like O followed by I-ORG become very unlikely
- **C.** It replaces word embeddings
- **D.** It makes training unsupervised

<details>
<summary>Answer</summary>

**B.** It learns transition scores between tags and decodes the best whole sequence, so invalid sequences like O followed by I-ORG become very unlikely

The CRF scores entire tag sequences with emission plus transition terms and normalizes over all sequences; Viterbi finds the best one.

- **A:** Tokenization happens before any tagger.
- **C:** The CRF sits on top of features or embeddings; it does not replace them.
- **D:** A CRF is trained on labeled tag sequences.

</details>

## 6. Multiple choice (hard)

A team reports 96% token accuracy for their NER model, but users say it misses about half the entities. What is the most likely explanation?

- **A.** The model is underfitting and needs more layers
- **B.** Token accuracy is dominated by easy O tokens; entity-level recall with exact span match is much lower
- **C.** The tokenizer is too fast
- **D.** Users are mislabeling entities

<details>
<summary>Answer</summary>

**B.** Token accuracy is dominated by easy O tokens; entity-level recall with exact span match is much lower

Most tokens are O. Report entity-level precision, recall, and F1 per type with exact span match, on entities unseen in training.

- **A:** Possibly true, but you can't conclude it before measuring the right metric.
- **C:** Tokenizer speed doesn't affect accuracy.
- **D:** Blaming users skips checking whether the metric matches what users care about.

</details>

## 7. Multiple choice (hard)

In the course script, adding the word itself as a feature lowered entity F1 on the test set from 0.73 to 0.56. Why?

- **A.** The word feature is buggy
- **B.** Test names never appear in training, so word-identity weights are useless there, and the model relied on them instead of shape and context
- **C.** More features always hurt logistic regression
- **D.** Viterbi cannot handle more features

<details>
<summary>Answer</summary>

**B.** Test names never appear in training, so word-identity weights are useless there, and the model relied on them instead of shape and context

The model memorized the training name list. Evaluate on unseen entities to measure generalization.

- **A:** The feature is computed correctly; it just doesn't transfer to new names.
- **C:** Extra features often help; here they encourage memorization because names don't recur.
- **D:** Viterbi only sees the per-tag probabilities; the feature count doesn't matter to it.

</details>

## 8. Multiple choice (medium)

Why can standard BIO not represent "Bank of America Stadium" as a FACILITY that contains the ORG "Bank of America"?

- **A.** BIO assigns exactly one tag per token, so a token can't belong to two overlapping spans
- **B.** BIO cannot represent multi-token entities
- **C.** BIO has no way to store the entity type
- **D.** BIO only works with transformers

<details>
<summary>Answer</summary>

**A.** BIO assigns exactly one tag per token, so a token can't belong to two overlapping spans

Nested entities need span-based models that score every candidate span, or one tagging pass per type.

- **B:** B followed by I tags represent multi-token entities.
- **C:** The type is part of every B-TYPE and I-TYPE tag.
- **D:** BIO works with any token classifier, including CRFs.

</details>

## 9. Multiple choice (easy)

What does `aggregation_strategy="simple"` do in the Hugging Face token-classification pipeline?

- **A.** Returns only the most confident entity
- **B.** Merges consecutive word pieces and B-/I- tokens of the same type into whole entity spans
- **C.** Lowercases the output
- **D.** Skips tokenization

<details>
<summary>Answer</summary>

**B.** Merges consecutive word pieces and B-/I- tokens of the same type into whole entity spans

Without aggregation you get one prediction per subword token and must merge them yourself.

- **A:** It returns all entities, grouped into spans.
- **C:** It does not change text case.
- **D:** The model still needs its subword tokenizer.

</details>

## 10. Multiple choice (hard)

You must redact names and account numbers from 50 million support tickets a day, on your own infrastructure. Which design fits best?

- **A.** Send every ticket to a hosted LLM with a redaction prompt
- **B.** Regex with checksums for account numbers plus a small fine-tuned encoder for names, tuned for high recall and measured with entity-level recall on unseen names
- **C.** A gazetteer of customer names only
- **D.** Token accuracy monitoring on a fixed test set

<details>
<summary>Answer</summary>

**B.** Regex with checksums for account numbers plus a small fine-tuned encoder for names, tuned for high recall and measured with entity-level recall on unseen names

Structured IDs suit regex; names need context. A small encoder meets cost and latency at this volume, and recall matters most for PII.

- **A:** About 580 calls per second is costly and slow, and sending PII to an external API may break policy.
- **C:** Names not in the list (new customers, family members) would leak.
- **D:** That's a metric, not a design, and token accuracy hides missed entities.

</details>

## 11. Select all that apply (medium)

When is a classical CRF or small encoder usually a better choice than prompting a large LLM for NER? Select all that apply.

- **A.** A strict latency budget of a few milliseconds per sentence
- **B.** Thousands of labeled examples and a stable label set
- **C.** Data that must stay on-premises on CPU-only machines
- **D.** No labeled data at all and a label set that changes weekly

<details>
<summary>Answer</summary>

**A, B, C**

Small trained models are fast, cheap, and accurate when labels exist and the ontology is stable. With no labels and a moving schema, LLM prompting is the faster start.

</details>
