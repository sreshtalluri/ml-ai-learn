<!-- GENERATED from bert-masked-language-modeling.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: BERT and masked language modeling

Covers the lesson [BERT and masked language modeling](../lessons/14-transformers/03-bert-masked-language-modeling.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/bert-masked-language-modeling/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What is BERT's main pretraining objective?

- **A.** Predict the next token from the tokens before it.
- **B.** Predict the original tokens at randomly selected, corrupted positions, using context on both sides.
- **C.** Classify whether two sentences are paraphrases.
- **D.** Reconstruct the input sentence in reverse order.

<details>
<summary>Answer</summary>

**B.** Predict the original tokens at randomly selected, corrupted positions, using context on both sides.

Masked language modeling selects about 15% of positions, corrupts them, and predicts the originals with a bidirectional encoder.

- **A:** That is the decoder (GPT) objective; with no causal mask an encoder could read the answer directly.
- **C:** Paraphrase classification is a fine-tuning task. BERT's second pretraining task was next-sentence prediction, later dropped.
- **D:** No major encoder is trained this way.

</details>

## 2. Calculation (medium)

Two positions are selected in a sentence. The model assigns the correct token probability 0.5 at the first and 0.2 at the second. What is the MLM loss (mean natural-log cross-entropy over selected positions)? (3 decimals)

<details>
<summary>Answer</summary>

**1.151** (within ±0.002)

$-\ln 0.5 = 0.6931$, $-\ln 0.2 = 1.6094$. Mean: $(0.6931 + 1.6094)/2 = 2.3025/2 = 1.1513$. Unselected positions have label $-100$ and add nothing.

</details>

## 3. Calculation (medium)

A batch has 32 sequences of 128 tokens, each including `[CLS]` and `[SEP]`. With selection rate 0.15, about how many positions per batch have a label (not $-100$)? (round to the nearest whole number)

<details>
<summary>Answer</summary>

**605** (within ±1)

Maskable tokens: $32 \times 126 = 4{,}032$. $0.15 \times 4{,}032 = 604.8 \approx 605$.

</details>

## 4. Multiple choice (medium)

Why are 10% of selected tokens replaced with a random token and 10% left unchanged, instead of always using `[MASK]`?

- **A.** To make training faster.
- **B.** `[MASK]` never appears at fine-tuning or inference time; mixing in real tokens means the model must build good representations of every visible token, not only of `[MASK]` positions.
- **C.** To increase the selection rate above 15%.
- **D.** Because `[MASK]` is not in the vocabulary.

<details>
<summary>Answer</summary>

**B.** `[MASK]` never appears at fine-tuning or inference time; mixing in real tokens means the model must build good representations of every visible token, not only of `[MASK]` positions.

If the model only ever predicted at `[MASK]`, its representations of ordinary tokens would matter less to the loss, creating a pretrain/fine-tune mismatch. Since it can't tell which visible tokens are corrupted, it has to model all of them.

- **A:** The rule doesn't change compute per step.
- **C:** The selection rate stays 15%; the rule only changes what the selected positions look like.
- **D:** `[MASK]` is a special token in the vocabulary.

</details>

## 5. Multiple choice (hard)

A team writes its own MLM training loop. Loss falls very fast to near zero, but the fine-tuned classifier is no better than one trained from scratch. Looking at the batch, labels equal the input ids at every position. What is wrong?

- **A.** The learning rate is too high.
- **B.** Labels are not set to $-100$ at unselected positions, so most of the loss is copying visible tokens, which is trivial.
- **C.** The model needs more layers.
- **D.** The 80/10/10 rule should be 90/5/5.

<details>
<summary>Answer</summary>

**B.** Labels are not set to $-100$ at unselected positions, so most of the loss is copying visible tokens, which is trivial.

With every position labelled, about 85% of targets are visible in the input, and a bidirectional encoder copies them. The loss looks great and teaches almost nothing. Labels must be $-100$ everywhere except the selected positions.

- **A:** A too-high learning rate would make loss unstable, not collapse it to near zero.
- **C:** Capacity is not the issue; the objective is trivial.
- **D:** The split matters far less than the missing loss mask.

</details>

## 6. Multiple choice (hard)

Semantic search built on cosine similarity of raw `bert-base-uncased` `[CLS]` vectors ranks unrelated documents highly. What is the best fix?

- **A.** Use a larger raw BERT model.
- **B.** Switch to an embedding model fine-tuned with a contrastive loss (a sentence-transformers style bi-encoder), and evaluate recall@k on labelled queries.
- **C.** Lower the similarity threshold.
- **D.** Increase the mask rate during pretraining.

<details>
<summary>Answer</summary>

**B.** Switch to an embedding model fine-tuned with a contrastive loss (a sentence-transformers style bi-encoder), and evaluate recall@k on labelled queries.

Raw BERT was not trained to place similar sentences close together, and its vectors cluster in a narrow cone. Contrastively fine-tuned embedding models are built for this.

- **A:** A bigger model with the same objective has the same problem.
- **C:** Thresholds can't fix a ranking that is wrong.
- **D:** The issue is the missing similarity training, not the pretraining mask rate.

</details>

## 7. Multiple choice (medium)

Which task is the worst fit for a BERT-style encoder?

- **A.** Named-entity recognition on support tickets.
- **B.** Reranking 50 retrieved passages for a query.
- **C.** Writing a 300-word reply to a customer email.
- **D.** Classifying messages into 20 intents at 5,000 requests per second.

<details>
<summary>Answer</summary>

**C.** Writing a 300-word reply to a customer email.

Open-ended generation needs a decoder (or encoder-decoder) that produces tokens autoregressively. Encoders excel at tagging, reranking, and classification.

- **A:** [object Object]
- **B:** Cross-encoder rerankers are fine-tuned BERT-style encoders.
- **D:** A small encoder classifies in one forward pass, ideal for high volume.

</details>

## 8. Select all that apply (medium)

Which statements are correct?

- **A.** RoBERTa showed that next-sentence prediction was not needed and dropped it.
- **B.** ELECTRA trains on every token by detecting which tokens a small generator replaced.
- **C.** BERT uses a causal attention mask like GPT.
- **D.** ModernBERT uses RoPE, GeGLU feed-forward layers, alternating local and global attention, and an 8,192-token context.
- **E.** The MLM loss is computed over all tokens in the sequence.

<details>
<summary>Answer</summary>

**A, B, D**

BERT has no causal mask, and the loss covers only selected positions.

- **C:** An encoder attends in both directions; that is the point of MLM.
- **E:** Only selected positions have labels; the rest are $-100$.

</details>

## 9. Reflection (hard)

You need PII detection (names, emails, account numbers) in 2 million chat messages per day, with a latency budget of 20 ms per message. Propose a model and explain why.

<details>
<summary>Answer</summary>

**Model answer.** Fine-tune a small BERT-style encoder (or ModernBERT) as a token classifier with BIO labels. Tagging needs one label per token using context on both sides, which an encoder gives in one forward pass with no generation loop; a 100M to 150M parameter model meets 20 ms on modest hardware. Add regex rules for structured patterns like emails and account numbers, use an LLM only to help label training data, and evaluate span-level precision and recall per entity type, weighting recall because missed PII is costly.

Token tagging under a latency budget is the canonical encoder use case.

</details>
