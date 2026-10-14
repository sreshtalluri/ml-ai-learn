---
title: CLIP and contrastive image-text pretraining
summary: Train an image encoder and a text encoder into one shared space with the symmetric contrastive loss, compute the similarity matrix, softmaxes, and loss by hand, use the result for zero-shot classification and retrieval, and compare softmax CLIP with sigmoid SigLIP.
skill: deep-learning
minutes: 45
prerequisites: [self-supervised-vision, vision-transformers, word-embeddings]
related: [vision-language-models, vector-search, rag-pipeline, ml-system-design]
adapted_from: [phases/12-multimodal-ai/02-clip-contrastive-pretraining]
---

# CLIP and contrastive image-text pretraining

> **Mental model.** Put photos and captions on the same map. CLIP trains two encoders, one for images and one for text, so that a photo lands right next to the sentence that describes it and far from every other sentence in the batch. Once the map exists, "classify this image" becomes "which of these sentences is closest?", for any list of sentences you care to write.

**You will learn to**
- Explain the dual-encoder design and why both outputs are normalised to unit length.
- Compute a batch's cosine similarity matrix, its row softmax (image to text) and column softmax (text to image), and the symmetric contrastive loss, by hand.
- Explain what the temperature does and why CLIP learns it.
- Run zero-shot classification with prompt templates, and explain when a linear probe or fine-tuning beats it.
- Compare CLIP's softmax loss with SigLIP's sigmoid loss, and name CLIP's known failure modes.

**Why it matters.** CLIP (OpenAI, 2021) learned from about 400 million web image-caption pairs with no hand labels and could classify ImageNet without seeing one ImageNet training label. Its descendants (OpenCLIP, SigLIP, SigLIP 2) are the image towers inside most open vision-language models, the engine of text-to-image search, and the text-conditioning backbone of many image generators. The loss is the same InfoNCE from [self-supervised vision](04-self-supervised-vision.md), applied across two modalities.

## 1. Intuition

**Two towers.** An image encoder (a [ViT](03-vision-transformers.md) or a ResNet) turns a picture into a vector. A text encoder (a small transformer) turns a caption into a vector of the same length. Both vectors are scaled to length 1, so their dot product is their **cosine similarity**: 1 means "pointing the same way", 0 means unrelated. This is the same [two-tower](../../glossary.md#two-tower-model) shape used in recommendation and retrieval.

**The game.** Take a batch of $N$ image-caption pairs. Compute every image against every caption: an $N \times N$ grid. The correct matches sit on the diagonal. Train both encoders so that:

- **reading across each row**, the image picks out its own caption among the $N$ captions (image to text), and
- **reading down each column**, the caption picks out its own image among the $N$ images (text to image).

Each direction is a softmax and a cross-entropy, with the diagonal as the right answer. The loss is the average of the two directions. Every other pair in the batch acts as a negative, so a batch of 32,768 (CLIP's size) gives each image 32,767 wrong captions to beat.

**Temperature.** Cosines live between $-1$ and $1$, which is too narrow for a decisive softmax. CLIP divides them by a temperature $\tau$ before the softmax. Small $\tau$ stretches small differences into big probability gaps; large $\tau$ flattens everything. CLIP learns $\tau$ during training, starting at 0.07, and clips it so it cannot go below 0.01.

**Zero-shot classification.** After training, write one caption per class ("a photo of a cat", "a photo of a dog", ...), embed them once, and for each image pick the caption with the highest cosine. No classifier is trained on these classes. Wording matters: "a photo of a {class}" usually beats the bare class name, and averaging the embeddings of many templates ("a photo of a", "a drawing of a", "a blurry photo of a") helps further.

> [!IMPORTANT]
> **Similarity is not truth.** A high cosine means "this caption is the closest of the ones you offered", not "this caption is correct". If none of your captions fits, CLIP still picks one. Treat the softmax over your prompt list as relative, and calibrate a threshold on labelled data before using it to reject images.

## 2. Visualization

<!-- lab:clip-similarity -->
![Three panels. Left: a 5 by 5 cosine similarity matrix between synthetic cat, dog, car, pizza, and tree images (rows) and captions (columns); the cat image scores 0.92 with its own caption but 0.96 with the dog caption. Middle: the row softmax at temperature 0.1, where the cat row puts 0.59 on dog and 0.41 on cat while the other rows are near 1 on the diagonal. Right: the symmetric contrastive loss falling from 0.217 to 0.018 over 30 training steps.](../../figures/clip-contrastive-pretraining.png)

*Interactive version: drag the temperature, switch between image-to-text and text-to-image softmaxes, press Train to pull matched pairs together, and rank captions for any image. [Open the lab](https://sreshtalluri.github.io/ml-ai-learn/labs/clip-similarity/).*
<!-- /lab -->

The embeddings are synthetic 5-dimensional vectors (axes roughly: animal, feline, vehicle, food, plant), small enough to read. The cat image starts out closer to the dog caption, so zero-shot classification gets 4 of 5 right. A few training steps fix it.

**Try it** (predict first, then check):
1. Switch to **text→image**. Does the caption "a photo of a cat" still pick the cat image? Why can the two directions disagree?
2. Drag $\tau$ from 0.1 down to 0.01. The cat row becomes almost one-hot on the *wrong* caption. Predict what happens to the loss, then check.
3. Load **Worked example (3 pairs)** and compare every cell with the hand calculation in section 3.
4. Press **Train** one step at a time and watch the cat row. How many steps until the cat image picks its own caption?

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $N$ | pairs in the batch | scalar |
| $f, g$ | image encoder, text encoder | |
| $u_i, v_j$ | unit-length embeddings of image $i$ and caption $j$ | $[D]$ |
| $C$ | cosine similarity matrix, $C_{ij} = u_i \cdot v_j$ | $[N, N]$ |
| $\tau$ | temperature | scalar |
| $P$ | row softmax of $C/\tau$ (image to text) | $[N, N]$ |
| $Q$ | column softmax of $C/\tau$ (text to image) | $[N, N]$ |

### Loss

```math
P_{ij} = \frac{e^{C_{ij}/\tau}}{\sum_{k} e^{C_{ik}/\tau}}, \qquad Q_{ij} = \frac{e^{C_{ij}/\tau}}{\sum_{k} e^{C_{kj}/\tau}}
```

```math
L_{\text{i2t}} = -\frac{1}{N}\sum_{i} \log P_{ii}, \qquad L_{\text{t2i}} = -\frac{1}{N}\sum_{j} \log Q_{jj}, \qquad L = \tfrac{1}{2}(L_{\text{i2t}} + L_{\text{t2i}})
```

If every softmax were uniform, $L = \ln N$: the chance level.

### Gradient

For the logits $S = C/\tau$, softmax cross-entropy gives the familiar "prediction minus target":

```math
\frac{\partial L}{\partial S_{ij}} = \frac{1}{2N}\left[(P_{ij} - \delta_{ij}) + (Q_{ij} - \delta_{ij})\right]
```

where $\delta_{ij}$ is 1 on the diagonal and 0 elsewhere. On the diagonal the gradient is negative (raise the matched similarity); off the diagonal it is positive (lower the confusing ones), in proportion to how much probability they stole. Through $C = UV^\top$ the image embeddings move along $\partial L/\partial U = \frac{1}{\tau}\frac{\partial L}{\partial S} V$.

### Zero-shot classification

```math
\hat{y}(x) = \arg\max_{k} \; u(x) \cdot v(\text{"a photo of a } k\text{"})
```

### SigLIP: a sigmoid per pair

```math
L_{\text{SigLIP}} = -\frac{1}{N}\sum_{i,j} \log \sigma\!\big(s_{ij}\,(t\,C_{ij} + b)\big), \qquad s_{ij} = +1 \text{ if } i = j, \; -1 \text{ otherwise}
```

Each pair is an independent yes/no question, with a learned scale $t$ and bias $b$ (initialised to 10 and $-10$, so the many negatives start out confidently "no"). No softmax ties a row together, so no device needs the whole $N \times N$ matrix at once. That makes very large batches cheaper to distribute, and SigLIP also works better than softmax CLIP at small batch sizes.

### Worked example: three pairs, $\tau = 0.1$

Unit vectors in 2-D:

- images: $u_1 = (1, 0)$, $u_2 = (0.6, 0.8)$, $u_3 = (0, 1)$
- captions: $v_1 = (0.8, 0.6)$, $v_2 = (0.6, 0.8)$, $v_3 = (0, 1)$

**Step 1: cosine matrix** (dot products):

| | $v_1$ | $v_2$ | $v_3$ |
|---|---|---|---|
| $u_1$ | 0.80 | 0.60 | 0.00 |
| $u_2$ | 0.48 + 0.48 = 0.96 | 0.36 + 0.64 = 1.00 | 0.80 |
| $u_3$ | 0.60 | 0.80 | 1.00 |

**Step 2: logits** $C/\tau$: row 1 is $(8, 6, 0)$, row 2 is $(9.6, 10, 8)$, row 3 is $(6, 8, 10)$.

**Step 3: row softmax (image to text).** Subtract each row's maximum first.

- Row 1: $e^{0}, e^{-2}, e^{-8} = 1, 0.1353, 0.0003$; sum $1.1357$; $P_{11} = 1/1.1357 = 0.8805$.
- Row 2: $e^{-0.4}, e^{0}, e^{-2} = 0.6703, 1, 0.1353$; sum $1.8057$; $P_{22} = 1/1.8057 = 0.5538$.
- Row 3: $e^{-4}, e^{-2}, e^{0} = 0.0183, 0.1353, 1$; sum $1.1536$; $P_{33} = 1/1.1536 = 0.8668$.

$L_{\text{i2t}} = -\frac{1}{3}(\ln 0.8805 + \ln 0.5538 + \ln 0.8668) = \frac{1}{3}(0.1273 + 0.5910 + 0.1430) = 0.2871$.

**Step 4: column softmax (text to image).**

- Column 1: logits $8, 9.6, 6$; relative to 9.6: $0.2019, 1, 0.0273$; sum $1.2292$; $Q_{11} = 0.2019/1.2292 = 0.1642$.
- Column 2: logits $6, 10, 8$; same numbers as row 3 in a different order; $Q_{22} = 1/1.1536 = 0.8668$.
- Column 3: logits $0, 8, 10$; $0.0000, 0.1353, 1$; sum $1.1354$; $Q_{33} = 0.8808$.

$L_{\text{t2i}} = \frac{1}{3}(1.8067 + 0.1430 + 0.1269) = 0.6922$.

**Step 5: symmetric loss** $L = (0.2871 + 0.6922)/2 = 0.4896$, against a chance level of $\ln 3 = 1.0986$.

Notice the asymmetry. Image 1 picks its own caption confidently (0.88), but caption 1 prefers image 2 (0.81 versus 0.16), because image 2 is close to everything. Training on both directions penalises such "hub" items.

**Temperature on the same numbers.** Image 2's row is $(0.35, 0.36, 0.29)$ at $\tau = 1$ (loss 0.924), $(0.37, 0.55, 0.08)$ at $\tau = 0.1$ (loss 0.490), and $(0.02, 0.98, 0.00)$ at $\tau = 0.01$. But at $\tau = 0.01$ caption 1 becomes almost certain of the *wrong* image, and the loss rises to 2.670. Very low temperatures punish every remaining confusion very hard; that is why CLIP clips $\tau$ at 0.01.

## 4. Implementation

The full script is [`code/25-computer-vision/clip_scratch.py`](../../code/25-computer-vision/clip_scratch.py) (NumPy only). It uses the same synthetic embeddings as the lab, reproduces the worked example, trains the embeddings for 30 steps, runs zero-shot classification, and compares the SigLIP loss.

### Build it from scratch

**Step 1: logits.** Normalise both sides so the dot product is a cosine, then divide by the temperature. One matrix multiply builds all $N^2$ comparisons.

```python
def clip_logits(img, txt, tau):
    """img, txt: [N, D] raw embeddings. Returns [N, N] logits S[i, j] = cos(img_i, txt_j) / tau."""
    return normalize(img) @ normalize(txt).T / tau
```

*Predict:* if you skip `normalize`, what happens to the logits of an image whose raw embedding is ten times longer than the others?

**Step 2: the symmetric loss.** The same matrix, softmaxed two ways. The correct answer is always the diagonal, so the "labels" are just `0, 1, ..., N-1`.

```python
def clip_loss(S):
    p_i2t = softmax(S, axis=1)                         # row i: which caption goes with image i?
    p_t2i = softmax(S, axis=0)                         # column j: which image goes with caption j?
    loss_i2t = -np.log(np.diag(p_i2t)).mean()         # correct answer is always the diagonal
    loss_t2i = -np.log(np.diag(p_t2i)).mean()
    return (loss_i2t + loss_t2i) / 2, loss_i2t, loss_t2i, p_i2t, p_t2i
```

*Predict:* the batch accidentally contains two identical captions for two different images. What is the best $P_{ii}$ either image can reach?

**Step 3: a training step on the embeddings.** In real CLIP the gradient flows back into both encoders. Here we update the embeddings directly to watch the geometry: softmax-minus-one-hot in both directions, chain rule through $S = IT^\top/\tau$, then re-normalise so everything stays on the unit sphere.

```python
def train_step(img, txt, tau=0.1, lr=0.05):
    """Gradient step on unit vectors, then re-normalise (projected gradient on the sphere)."""
    I, T = normalize(img), normalize(txt)
    n = len(I)
    _, _, _, p, q = clip_loss(I @ T.T / tau)
    dS = ((p - np.eye(n)) + (q - np.eye(n))) / (2 * n)  # dLoss/dS: softmax minus one-hot, both directions
    dI = dS @ T / tau                                  # chain rule through S = I T^T / tau
    dT = dS.T @ I / tau
    return normalize(I - lr * dI), normalize(T - lr * dT)
```

*Predict:* which off-diagonal cell gets the largest positive gradient at step 0 in the lab data? (Look at where the most probability is misplaced.)

**Step 4: SigLIP for comparison.** One sigmoid per cell, no softmax across the row or column.

```python
def siglip_loss(cos, t=10.0, b=-10.0):
    """cos: [N, N] cosines. Positive pairs on the diagonal, negatives everywhere else."""
    z = t * cos + b
    sign = 2 * np.eye(len(cos)) - 1                    # +1 on the diagonal, -1 elsewhere
    return float(np.log1p(np.exp(-sign * z)).sum() / len(cos))
```

*Predict:* with $t = 10$ and $b = -10$, a negative pair with cosine 0 contributes $\ln(1 + e^{-10})$, almost nothing. What does a positive pair with cosine 0.9 contribute?

### Use the library

Pretrained CLIP-family models are available from Hugging Face `transformers` and `open_clip`. Zero-shot classification is a few lines.

```python
import torch
from transformers import AutoModel, AutoProcessor            # pip install transformers torch pillow

name = "openai/clip-vit-base-patch32"                        # or a SigLIP / SigLIP 2 checkpoint
model, proc = AutoModel.from_pretrained(name).eval(), AutoProcessor.from_pretrained(name)
labels = ["cat", "dog", "car", "pizza", "tree"]
inputs = proc(text=[f"a photo of a {c}" for c in labels], images=pil_image, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model(**inputs)
probs = out.logits_per_image.softmax(dim=-1)                 # [1, 5]: CLIP applies the learned 1/tau for you
print(labels[probs.argmax()])
```

What the library adds: trained encoders, the learned temperature (exposed as `logit_scale`), tokenisation and image preprocessing matched to the checkpoint, and, for SigLIP, sigmoid-based logits. Note that SigLIP checkpoints are meant to be read with a sigmoid per caption, not a softmax across captions. For retrieval at scale, embed the corpus once, store the normalised vectors in an [ANN index](../16-rag/02-vector-search.md), and embed only the query at request time.

### Run it

```bash
uv run code/25-computer-vision/clip_scratch.py
```

Key lines of output (synthetic embeddings):

```text
loss_i2t = 0.2870   loss_t2i = 0.6921   symmetric loss = 0.4896
chance level ln 3 = 1.0986
tau 0.01: image 2 row = [0.018, 0.982, 0.0]  loss = 2.6697
loss = 0.2167 (i2t 0.2316, t2i 0.2018)
zero-shot (best caption per image): ['cat->dog', 'dog->dog', 'car->car', 'pizza->pizza', 'tree->tree'] accuracy 80%
step  5: loss 0.1103  P(cat image -> 'cat' caption) = 0.676  accuracy 100%
step 30: loss 0.0178  P(cat image -> 'cat' caption) = 0.958  accuracy 100%
before training 1.0084   after 30 steps 0.7784
```

(The last decimal of the first line differs from the hand calculation because the script does not round intermediate steps.)

## 5. Engineering

**Use cases.** Text-to-image and image-to-image search; zero-shot or few-shot classification and tagging; content moderation filters; deduplication and clustering of image collections; scoring how well a generated image matches its prompt; and the image encoder of vision-language models ([next lesson](06-vision-language-models.md)).

**Zero-shot, linear probe, or fine-tune.** Zero-shot needs no labels and keeps the model general. A linear probe on frozen image embeddings, with even a few labelled examples per class, usually beats zero-shot on a specific domain. Full fine-tuning gains the most in-domain but can erode zero-shot ability on everything else; if you need both, fine-tune with a small learning rate, use weight averaging with the original model, or keep the original for open-ended queries.

**Prompts.** Match the template to the data ("a satellite photo of {}", "a product photo of {}"), average several templates, and use class names the web would use. Prompt choice can move zero-shot accuracy by several points, so evaluate it like any other hyperparameter.

**Training your own.** Data quality dominates. Filter pairs by an existing model's similarity score, deduplicate, and balance concepts. Softmax CLIP needs very large batches spread across devices, with embeddings gathered for the full $N \times N$ matrix; SigLIP's per-pair loss distributes more cheaply. Most teams fine-tune an open checkpoint (OpenCLIP, SigLIP 2) instead of pretraining.

**Choosing a model in 2026.** SigLIP 2 (2025) is the common default for new open systems: it adds multilingual training, better dense (per-patch) features, and a variant that accepts native aspect ratios and resolutions. OpenCLIP checkpoints remain widely used for retrieval. Check the license and the training data's languages and domains against yours.

> [!WARNING]
> **Failure modes.** CLIP-style models are weak at counting, spatial relations ("left of"), negation, and binding attributes to the right object ("a red cube and a blue sphere" versus the reverse); they behave like bags of concepts. They can be fooled by text written in the image (a label reading "iPod" stuck on an apple). They absorb web biases, including harmful associations, so audit before using them for decisions about people. Zero-shot scores always pick *some* caption, even when none applies.

### Common mistakes

- Forgetting to normalise embeddings, so "cosine" scores grow with vector length.
- Comparing raw cosine values across different models; only rankings within one model are meaningful.
- Treating the zero-shot softmax over your prompt list as a calibrated probability.
- Using a softmax across captions with a SigLIP checkpoint that was trained with sigmoids.
- Mixing images preprocessed with ImageNet statistics into a model trained with CLIP's normalisation.
- Indexing a corpus with one checkpoint and embedding queries with another.

## 6. Knowledge check

<!-- quiz:clip-contrastive-pretraining -->
**[Take the CLIP quiz](../../quizzes/clip-contrastive-pretraining.md)**: similarity matrices, temperature, the symmetric loss, zero-shot, and SigLIP.
<!-- /quiz -->

**Practice exercise.** Two pairs, $\tau = 0.5$. Cosines: $C_{11} = 0.9$, $C_{12} = 0.5$, $C_{21} = 0.3$, $C_{22} = 0.7$. Compute $P_{11}$, $P_{22}$, $L_{\text{i2t}}$, $Q_{11}$, $Q_{22}$, $L_{\text{t2i}}$, and the symmetric loss.

<details>
<summary>Solution</summary>

Logits: row 1 $(1.8, 1.0)$, row 2 $(0.6, 1.4)$.

- $P_{11} = 1/(1 + e^{-0.8}) = 1/1.4493 = 0.690$. $P_{22} = 1/(1 + e^{-0.8}) = 0.690$. $L_{\text{i2t}} = -\ln 0.690 = 0.371$.
- Column 1 logits $(1.8, 0.6)$: $Q_{11} = 1/(1 + e^{-1.2}) = 1/1.3012 = 0.769$. Column 2 logits $(1.0, 1.4)$: $Q_{22} = 1/(1 + e^{-0.4}) = 1/1.6703 = 0.599$. $L_{\text{t2i}} = \frac{1}{2}(0.263 + 0.513) = 0.388$.
- $L = (0.371 + 0.388)/2 = 0.380$, against a chance level of $\ln 2 = 0.693$.

</details>

**Implementation challenge.** Add *prompt ensembling* to the script: give each class three caption vectors (the original plus two small random perturbations, simulating different templates), average and re-normalise them per class, and compare zero-shot accuracy and the cat row's probabilities with single-caption classification.

<details>
<summary>Solution sketch</summary>

```python
rng = np.random.default_rng(1)
variants = [CAPTIONS + 0.15 * rng.standard_normal(CAPTIONS.shape) for _ in range(2)]
ensemble = normalize(normalize(CAPTIONS) + sum(normalize(v) for v in variants))
P = softmax(normalize(IMAGES) @ ensemble.T / 0.1, axis=1)
print(P.argmax(1), P[0].round(3))
```

Averaging several noisy descriptions of the same class cancels template-specific noise and keeps the shared direction, which is why ensembling usually helps a little. With random perturbations the result depends on the seed. Report the change over several seeds, not one.

</details>

## Summary

- CLIP trains an image encoder and a text encoder so matched pairs have high cosine similarity and every other pair in the batch has low similarity.
- The loss is cross-entropy over the rows (image to text) and the columns (text to image) of the $N \times N$ similarity matrix divided by a learned temperature; chance level is $\ln N$.
- Temperature controls how hard near misses are punished; CLIP learns it and clips it at 0.01.
- Zero-shot classification embeds one prompt per class and picks the nearest; linear probes and fine-tuning trade generality for in-domain accuracy.
- SigLIP replaces the batch softmax with independent sigmoids, which distributes better and is the basis of many 2025–2026 vision towers.
- CLIP-style models are weak at counting, spatial relations, negation, and attribute binding, and they inherit web biases.

**Next:** [Vision-language models](06-vision-language-models.md)

**Related:** [Self-supervised vision](04-self-supervised-vision.md) · [Vector search](../16-rag/02-vector-search.md) · [ML system design](../22-ml-system-design/01-ml-system-design.md) · [Word embeddings](../09-classical-nlp/02-word-embeddings.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [CLIP and Contrastive Vision-Language Pretraining](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/02-clip-contrastive-pretraining).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Walk me through CLIP's training objective for one batch.</strong></summary>

Take N image-caption pairs. Encode the images with the image tower and the captions with the text tower, and L2-normalise both so dot products are cosines. Multiply to get an N by N similarity matrix and divide by a learned temperature, initially 0.07. The matching pairs are the diagonal. Apply a softmax across each row and take the cross-entropy against the diagonal: that is the image-to-text loss, where each image must pick its caption from N. Apply a softmax down each column for the text-to-image loss. Average the two. Every off-diagonal pair is a negative, so batch size is the number of choices; CLIP used 32,768. The gradient on each logit is the softmax probability minus the one-hot target, so matched pairs are pulled together and the confusing negatives pushed apart in proportion to the probability they stole.

</details>

<details>
<summary><strong>What does the temperature do, and what goes wrong if it is too high or too low?</strong></summary>

Cosines lie between −1 and 1, which would give an almost flat softmax over thousands of candidates. Dividing by τ stretches them. At high τ, around 1, probabilities stay near uniform, the loss sits close to ln N, and gradients are weak and spread across all negatives, so learning is slow. At very low τ the softmax becomes nearly one-hot. Correct pairs cost almost nothing, but any remaining confusion is punished hard and gradients concentrate on a few hardest negatives, which can destabilise training or overfit to noisy pairs. In the worked example, one confusable caption makes the loss rise from 0.49 at τ = 0.1 to 2.67 at τ = 0.01. CLIP learns τ in log space and clips it so 1/τ never exceeds 100, which lets the model sharpen as its embeddings improve without blowing up.

</details>

<details>
<summary><strong>CLIP versus SigLIP: why would you choose the sigmoid loss?</strong></summary>

CLIP's softmax normalises over the whole row and column, so every device needs every other device's embeddings to compute its loss. That means an all-gather of the full batch and an N by N matrix, which gets expensive at very large batch sizes. SigLIP treats each image-caption pair as an independent binary question: is this a match? It uses a sigmoid with a learned scale and bias, initialised at 10 and −10 so the many negatives start out as confident "no" answers. There is no global normalisation, so devices can compute their blocks and swap text embeddings in chunks with less memory. The SigLIP paper also found it works better than softmax at small and moderate batch sizes. The trade-off is that scores are per-pair probabilities rather than a ranking over a set, so read SigLIP checkpoints with a sigmoid. SigLIP 2 added multilingual data, dense features, and native-resolution inputs.

</details>

<details>
<summary><strong>Your CLIP-based product search returns red shirts for "a blue shirt with red stripes" and fails on "not leather". Explain and fix it.</strong></summary>

Both come from how contrastive pretraining shapes the embedding. Web captions rarely require fine distinctions, so one pooled vector per image and per text behaves like a bag of concepts. It knows red, blue, shirt, and stripes are present but binds attributes to objects weakly, and it nearly ignores negation, because "not leather" sits close to "leather". Fixes, in order of cost: parse the query and use structured filters for attributes and negations from catalogue metadata, where you have it. Retrieve the top 100 with CLIP for recall, then rerank with a model that reads image and text jointly, such as a cross-attention reranker or a vision-language model asked a yes-or-no question. Fine-tune on your own catalogue with hard negatives, pairing "blue shirt, red stripes" against "red shirt, blue stripes". Then measure with a labelled query set that includes these compositional cases.

</details>
