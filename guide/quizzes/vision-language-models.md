<!-- GENERATED from vision-language-models.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Vision-language models

Covers the lesson [Vision-language models](../lessons/25-computer-vision/06-vision-language-models.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/vision-language-models/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

In the first (alignment) stage of training a ViT-projector-LLM model, which component is usually trained?

- **A.** All components together
- **B.** The vision encoder only
- **C.** The LLM only
- **D.** The projector only, with the vision encoder and LLM frozen

<details>
<summary>Answer</summary>

**D.** The projector only, with the vision encoder and LLM frozen

The projector is small (tens of millions of parameters) and learns to map vision features into the LLM's embedding space. Later stages unfreeze more.

- **A:** That comes later, if at all.
- **B:** The vision encoder is usually frozen early to keep its pretrained features.
- **C:** The LLM is frozen in alignment.
- **D:** Correct.

</details>

## 2. Multiple choice (medium)

Why do most VLMs use a CLIP- or SigLIP-trained vision encoder rather than an ImageNet-supervised one?

- **A.** They have more parameters
- **B.** ResNets cannot produce tokens
- **C.** Their features were trained to align with text, so a small projector can map them into the LLM's word space
- **D.** They are always faster

<details>
<summary>Answer</summary>

**C.** Their features were trained to align with text, so a small projector can map them into the LLM's word space

Contrastive image-text pretraining shapes features that are already language-compatible.

- **A:** Size is not the reason.
- **B:** Any CNN feature map can be flattened into tokens.
- **C:** Correct.
- **D:** Speed depends on the specific model.

</details>

## 3. Calculation (easy)

A VLM's vision encoder uses 14-pixel patches at 448 by 448 pixels with no token merging. How many image tokens does one image produce?

<details>
<summary>Answer</summary>

**1024**

$448/14 = 32$ per side, $32^2 = 1024$.

</details>

## 4. Calculation (medium)

A model tiles a large image into 4 tiles of 336 pixels plus one thumbnail (14-pixel patches) and then applies 2 by 2 token merging. How many image tokens result?

<details>
<summary>Answer</summary>

**720**

Each tile has $24^2 = 576$ tokens; five views give $2880$; merging divides by 4, giving $720$.

</details>

## 5. Calculation (medium)

How many parameters (in millions, one decimal) does a two-layer MLP projector from 1024 to 4096 to 4096 have, including biases?

<details>
<summary>Answer</summary>

**21** (within ±0.1)

$1024 \times 4096 + 4096 + 4096 \times 4096 + 4096 = 20{,}979{,}712 \approx 21.0$ million.

</details>

## 6. Multiple choice (medium)

Compared with an MLP projector, what does a Q-Former (as in BLIP-2) trade?

- **A.** It keeps every patch token but needs a larger LLM
- **B.** A fixed small number of query tokens per image (cheap context) at the risk of losing fine detail such as small text
- **C.** It removes the vision encoder
- **D.** It makes images free in the context window

<details>
<summary>Answer</summary>

**B.** A fixed small number of query tokens per image (cheap context) at the risk of losing fine detail such as small text

Learned queries cross-attend to patch features and summarise them into, for example, 32 tokens regardless of resolution.

- **A:** That describes the MLP projector.
- **B:** Correct.
- **C:** The Q-Former reads features from a vision encoder.
- **D:** The 32 query tokens still occupy the context.

</details>

## 7. Multiple choice (hard)

A VLM confidently describes objects that are not in the image. Which evaluation best measures this?

- **A.** Perplexity on captions
- **B.** Average answer length
- **C.** Yes/no questions about objects that are present and absent in labelled images, reporting the false-yes rate on absent objects separately
- **D.** Token generation latency

<details>
<summary>Answer</summary>

**C.** Yes/no questions about objects that are present and absent in labelled images, reporting the false-yes rate on absent objects separately

Absent-object probes (POPE-style) isolate object hallucination; aggregate accuracy hides it.

- **A:** Perplexity measures fluency, not grounding.
- **B:** Length says nothing about correctness.
- **C:** Correct.
- **D:** Latency is unrelated.

</details>

## 8. Multiple choice (hard)

After porting a VLM to a new serving stack, answers ignore the image and later text seems garbled. The prompt has 576 `<image>` placeholders but the new processor produces 729 image features. What is the most likely cause and fix?

- **A.** The LLM is too small; use a larger one
- **B.** Preprocessing (resolution or patch size) differs from the checkpoint's, so placeholder and feature counts disagree; match the processor configuration and fail loudly on any mismatch
- **C.** Sampling temperature is too high
- **D.** The projector needs retraining

<details>
<summary>Answer</summary>

**B.** Preprocessing (resolution or patch size) differs from the checkpoint's, so placeholder and feature counts disagree; match the processor configuration and fail loudly on any mismatch

729 is $27^2$ (378 pixels with 14-pixel patches) versus 576, which is $24^2$ (336 pixels). Use the checkpoint's own processor and assert that counts match.

- **A:** Model size does not explain a count mismatch.
- **B:** Correct.
- **C:** Temperature does not change the input sequence.
- **D:** The weights are fine; the inputs are wrong.

</details>

## 9. Multiple choice (medium)

A warehouse needs an exact count of boxes on each pallet photo, plus a one-sentence damage note. Which design is best?

- **A.** Ask a VLM for both
- **B.** A fine-tuned detector for the count, and a VLM (optionally given the detections) for the damage note
- **C.** CLIP zero-shot with prompts for each possible count
- **D.** A segmentation model for the note

<details>
<summary>Answer</summary>

**B.** A fine-tuned detector for the count, and a VLM (optionally given the detections) for the damage note

Detectors count reliably and are easy to evaluate; VLMs are weak at counting but good at short descriptions.

- **A:** VLM counting is unreliable, especially for many similar objects.
- **B:** Correct.
- **C:** CLIP is poor at counting.
- **D:** Segmentation produces masks, not sentences.

</details>
