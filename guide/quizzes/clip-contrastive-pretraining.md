<!-- GENERATED from clip-contrastive-pretraining.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: CLIP and contrastive image-text pretraining

Covers the lesson [CLIP and contrastive image-text pretraining](../lessons/25-computer-vision/05-clip-contrastive-pretraining.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/clip-contrastive-pretraining/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

In CLIP's $N \times N$ similarity matrix for one batch, where are the correct image-caption pairs?

- **A.** On the diagonal
- **B.** In the first row
- **C.** Wherever the cosine is highest
- **D.** Off the diagonal

<details>
<summary>Answer</summary>

**A.** On the diagonal

Image $i$ was paired with caption $i$, so the targets for both softmaxes are the diagonal entries.

- **A:** Correct.
- **B:** The first row holds image 1 against every caption.
- **C:** Early in training the highest cosine is often wrong; the label comes from the pairing.
- **D:** Off-diagonal cells are the negatives.

</details>

## 2. Multiple choice (medium)

Why does CLIP average an image-to-text loss (row softmax) and a text-to-image loss (column softmax)?

- **A.** The two are always equal, so averaging is harmless
- **B.** Each direction catches different confusions; a "hub" image close to many captions can win its row yet steal other captions' columns
- **C.** It halves the compute
- **D.** Only the text-to-image loss is used at inference

<details>
<summary>Answer</summary>

**B.** Each direction catches different confusions; a "hub" image close to many captions can win its row yet steal other captions' columns

In the lesson's worked example image 1 picks its caption with 0.88, but caption 1 prefers image 2 with 0.81. Training both directions penalises such asymmetries.

- **A:** They differ (0.287 versus 0.692 in the worked example).
- **B:** Both softmaxes come from the same matrix; no compute is saved.
- **C:** Correct.
- **D:** Both directions are used in practice (search and classification).

</details>

## 3. Calculation (medium)

An image's cosines with three captions are $(0.96, 1.00, 0.80)$, and its own caption is the second. At $\tau = 0.1$, what probability does the row softmax give its own caption? (Four decimals.)

<details>
<summary>Answer</summary>

**0.5538** (within ±0.0005)

Logits $(9.6, 10, 8)$. Relative to 10 the exponentials are $e^{-0.4} = 0.6703$, $1$, $e^{-2} = 0.1353$; sum $1.8057$; probability $1/1.8057 = 0.5538$.

</details>

## 4. Calculation (hard)

Two pairs, $\tau = 0.5$, cosines $C_{11} = 0.9$, $C_{12} = 0.5$, $C_{21} = 0.3$, $C_{22} = 0.7$. Compute the symmetric contrastive loss. (Three decimals.)

<details>
<summary>Answer</summary>

**0.38** (within ±0.003)

Rows: logits $(1.8, 1.0)$ and $(0.6, 1.4)$; both diagonal probabilities are $1/(1 + e^{-0.8}) = 0.690$, so $L_{\text{i2t}} = 0.371$. Columns: $(1.8, 0.6)$ gives $0.769$; $(1.0, 1.4)$ gives $0.599$; $L_{\text{t2i}} = (0.263 + 0.513)/2 = 0.388$. Average $0.380$.

</details>

## 5. Multiple choice (medium)

What happens to CLIP's softmax as the temperature $\tau$ decreases toward 0.01?

- **A.** Probabilities become more uniform
- **B.** Probabilities become nearly one-hot, so remaining confusions are punished very hard
- **C.** Nothing; temperature only affects inference
- **D.** The cosines change

<details>
<summary>Answer</summary>

**B.** Probabilities become nearly one-hot, so remaining confusions are punished very hard

Dividing by a smaller $\tau$ stretches cosine gaps. Correct pairs become cheap and any confusion becomes very expensive (loss 2.67 at $\tau = 0.01$ in the worked example).

- **A:** That happens as $\tau$ increases.
- **B:** Correct.
- **C:** CLIP learns $\tau$ during training.
- **D:** Temperature scales logits; the cosines stay the same.

</details>

## 6. Calculation (easy)

With a batch of 1024 pairs, what symmetric contrastive loss does a model with uniform softmaxes get? (Natural log, two decimals.)

<details>
<summary>Answer</summary>

**6.93** (within ±0.01)

Both directions give $\ln 1024 = 6.931$, so the average is $6.93$.

</details>

## 7. Multiple choice (easy)

How does CLIP classify an image into classes it was never trained on?

- **A.** It fine-tunes a new head for the classes
- **B.** It embeds a text prompt per class (such as "a photo of a dog") and picks the class whose prompt embedding is most similar to the image embedding
- **C.** It looks up the class in an ImageNet table
- **D.** It generates a caption and parses it

<details>
<summary>Answer</summary>

**B.** It embeds a text prompt per class (such as "a photo of a dog") and picks the class whose prompt embedding is most similar to the image embedding

Classification becomes retrieval among the prompts you supply; no weights change.

- **A:** That would be a linear probe or fine-tuning, not zero-shot.
- **B:** Correct.
- **C:** CLIP has no fixed label table.
- **D:** CLIP cannot generate text.

</details>

## 8. Multiple choice (medium)

What is the main practical advantage of SigLIP's sigmoid loss over CLIP's softmax loss?

- **A.** It needs no text encoder
- **B.** Each pair is an independent binary decision, so there is no normalisation across the whole batch; this distributes more cheaply and works well at smaller batches too
- **C.** It removes the need for negatives
- **D.** It produces captions

<details>
<summary>Answer</summary>

**B.** Each pair is an independent binary decision, so there is no normalisation across the whole batch; this distributes more cheaply and works well at smaller batches too

The softmax couples every entry in a row and column, which requires the full matrix. Sigmoids do not.

- **A:** SigLIP keeps two towers.
- **B:** Correct.
- **C:** Off-diagonal pairs are still negatives, with label $-1$.
- **D:** Neither model generates text.

</details>

## 9. Multiple choice (hard)

CLIP-based product search returns red shirts for "a blue shirt with red stripes" and leather bags for "bag, not leather". What is the best fix?

- **A.** Lower the temperature
- **B.** Retrieve candidates with CLIP for recall, then rerank with a model that reads image and text jointly, and handle attributes and negations with structured filters where catalogue metadata exists
- **C.** Use a larger batch at inference
- **D.** Normalise the query twice

<details>
<summary>Answer</summary>

**B.** Retrieve candidates with CLIP for recall, then rerank with a model that reads image and text jointly, and handle attributes and negations with structured filters where catalogue metadata exists

Pooled dual-encoder embeddings bind attributes weakly and largely ignore negation. Joint rerankers and metadata filters handle these cases.

- **A:** Temperature scales scores; it does not change the ranking.
- **B:** Correct.
- **C:** Batch size at inference does not change the embeddings.
- **D:** Normalisation is idempotent.

</details>

## 10. Multiple choice (medium)

You want to flag unsafe images using CLIP zero-shot with prompts "a safe photo" and "an unsafe photo". What is the most important step before launch?

- **A.** Nothing; the softmax gives a calibrated probability
- **B.** Build a labelled evaluation set, choose prompts and a threshold on it, and audit errors across groups of people and content types
- **C.** Add more prompts until the probabilities look confident
- **D.** Switch to a lower temperature

<details>
<summary>Answer</summary>

**B.** Build a labelled evaluation set, choose prompts and a threshold on it, and audit errors across groups of people and content types

Zero-shot scores are relative to the prompts offered and not calibrated; CLIP also inherits web biases. Decisions need a measured threshold and an audit.

- **A:** The softmax over two prompts always picks one; it is not a calibrated probability.
- **B:** Correct.
- **C:** Confidence is not accuracy.
- **D:** That makes outputs more extreme, not more correct.

</details>
