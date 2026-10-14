<!-- GENERATED from self-supervised-vision.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Self-supervised vision

Covers the lesson [Self-supervised vision](../lessons/25-computer-vision/04-self-supervised-vision.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/self-supervised-vision/) grades these interactively and tracks a review queue.

## 1. Multiple choice (medium)

Why did SimCLR need very large batches (thousands of images)?

- **A.** The other views in the batch are the negatives; small batches give too few negatives for a strong contrastive signal
- **B.** Batch norm requires it
- **C.** The loss divides by batch size
- **D.** Large batches are always faster

<details>
<summary>Answer</summary>

**A.** The other views in the batch are the negatives; small batches give too few negatives for a strong contrastive signal

Each view competes against $2N - 2$ negatives. More negatives mean more hard negatives and a sharper signal. MoCo's queue decouples the negative count from batch size.

- **A:** Correct.
- **B:** Batch norm works at ordinary batch sizes.
- **C:** The averaging is not the reason.
- **D:** Speed is not the motivation.

</details>

## 2. Multiple choice (medium)

What keeps DINO, which uses no negatives, from collapsing to a constant output?

- **A.** A large batch size
- **B.** The learning-rate schedule
- **C.** Augmentation alone
- **D.** Centring the teacher's output (subtract a running mean) together with sharpening it (low teacher temperature)

<details>
<summary>Answer</summary>

**D.** Centring the teacher's output (subtract a running mean) together with sharpening it (low teacher temperature)

Centring stops one dimension from winning for every input; sharpening stops the output from going uniform. Each alone collapses in a different way.

- **A:** DINO works without huge batches.
- **B:** Schedules don't prevent the trivial solution.
- **C:** Augmentations alone cannot rule out a constant output.
- **D:** Correct.

</details>

## 3. Multiple choice (medium)

MAE masks 75 percent of image patches; BERT masks 15 percent of text tokens. Why the difference?

- **A.** 75 percent is arbitrary
- **B.** BERT requires 15 percent by design
- **C.** Image patches are highly redundant, so a low ratio can be solved by local interpolation; a high ratio forces semantic understanding
- **D.** Text tokens are larger than image patches

<details>
<summary>Answer</summary>

**C.** Image patches are highly redundant, so a low ratio can be solved by local interpolation; a high ratio forces semantic understanding

The mask ratio should match the information density of the modality. Neighbouring patches nearly determine each other.

- **A:** It was chosen by experiment for this reason.
- **B:** BERT's ratio was also a tuned choice.
- **C:** Correct.
- **D:** Size in bytes is not the point.

</details>

## 4. Multiple choice (easy)

In a linear probe evaluation, what is trained?

- **A.** A full MLP head
- **B.** The whole encoder
- **C.** The positional embeddings
- **D.** A single linear classifier on top of the frozen encoder's features

<details>
<summary>Answer</summary>

**D.** A single linear classifier on top of the frozen encoder's features

Freezing the encoder isolates feature quality from further training.

- **A:** An MLP adds nonlinear capacity, which blurs the measurement.
- **B:** That is fine-tuning.
- **C:** Not trained in a probe.
- **D:** Correct.

</details>

## 5. Multiple choice (hard)

Why does MAE use a large encoder on visible patches only and a small decoder on all positions?

- **A.** The decoder needs its own backbone
- **B.** The encoder never processes mask tokens, so its cost scales with the 25 percent visible patches; the small decoder is discarded after pretraining
- **C.** Mask tokens confuse attention
- **D.** To reduce memory for the dataset

<details>
<summary>Answer</summary>

**B.** The encoder never processes mask tokens, so its cost scales with the 25 percent visible patches; the small decoder is discarded after pretraining

Encoder attention cost falls by about $1/(1-r)^2$, roughly 16 times at $r = 0.75$, making pretraining much cheaper.

- **A:** The decoder is a few light transformer blocks.
- **B:** Correct.
- **C:** Mask tokens are fine in the decoder; the point is cost.
- **D:** Dataset storage is unaffected.

</details>

## 6. Calculation (medium)

SimCLR with $N = 64$ images per batch. What InfoNCE loss would uniform probabilities give? (Natural log, three decimals.)

<details>
<summary>Answer</summary>

**4.844** (within ±0.002)

$2N = 128$ views; each has $127$ candidates. Uniform probabilities give $\ln 127 = 4.844$.

</details>

## 7. Calculation (hard)

A view has logits (cosine divided by temperature) of 1.6 for its twin and 0 and 1.2 for the two negatives. What is its InfoNCE term? (Three decimals.)

<details>
<summary>Answer</summary>

**0.627** (within ±0.002)

$e^{1.6} = 4.953$, $e^0 = 1$, $e^{1.2} = 3.320$; sum $9.273$; probability $0.534$; $-\ln 0.534 = 0.627$.

</details>

## 8. Multiple choice (hard)

Your contrastive model matches views almost perfectly during pretraining, but its nearest neighbours are always images from the same camera, regardless of content. What is happening?

- **A.** The temperature is too high
- **B.** Shortcut learning; a camera signature shared by both views (colour cast, watermark, compression artefacts) lets the model match views without understanding content
- **C.** The model has collapsed
- **D.** The batch is too large

<details>
<summary>Answer</summary>

**B.** Shortcut learning; a camera signature shared by both views (colour cast, watermark, compression artefacts) lets the model match views without understanding content

Augmentations must destroy cues you don't want learned. Add colour and compression augmentation, crop out watermarks, or sample negatives from the same camera.

- **A:** High temperature makes matching harder, not easier.
- **B:** Correct.
- **C:** Collapse would map all images to the same point, not cluster by camera.
- **D:** More negatives would not remove the shared cue.

</details>

## 9. Multiple choice (medium)

You have 2 million unlabelled microscopy images and 3,000 labelled ones. What is the most sensible plan?

- **A.** Train a supervised model on the 3,000 images from scratch
- **B.** Start from a public self-supervised checkpoint, continue self-supervised pretraining on the 2 million images, then probe or fine-tune on the 3,000
- **C.** Use zero-shot CLIP with class names
- **D.** Label all 2 million images first

<details>
<summary>Answer</summary>

**B.** Start from a public self-supervised checkpoint, continue self-supervised pretraining on the 2 million images, then probe or fine-tune on the 3,000

Domain-adaptive self-supervised pretraining uses the unlabelled pile; the labelled set is then enough for a probe or light fine-tune.

- **A:** 3,000 images is little for a deep network from scratch.
- **B:** Correct.
- **C:** Microscopy is far from web captions; zero-shot is likely weak.
- **D:** Too expensive, and unnecessary.

</details>
