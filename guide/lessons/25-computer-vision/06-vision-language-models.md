---
title: Vision-language models
summary: Explain how a vision encoder, a projector, and a language model combine into a VLM, trace image tokens into the LLM's sequence, budget image tokens and KV-cache memory, describe the training stages, and evaluate and monitor visual hallucination.
skill: deep-learning
minutes: 40
prerequisites: [clip-contrastive-pretraining, vision-transformers, tokenization-and-pretraining]
related: [fine-tuning-in-practice, llm-evaluation, llm-inference, tool-use-and-agents]
adapted_from: [phases/04-computer-vision/25-vision-language-models]
---

# Vision-language models

> **Mental model.** A vision-language model is a language model that has been given a translator for pictures. A vision encoder turns the image into a few hundred patch vectors; a small projector translates each one into the language model's "word vector" format; then those translated vectors are pasted into the prompt where the image goes. The language model reads them like unusual words and answers in text.

**You will learn to**
- Name the three parts of a ViT-projector-LLM model and say what each contributes and which ones are usually trained.
- Trace how `<image>` placeholder tokens are replaced by projected image features, with shapes.
- Compute how many tokens an image costs at a given resolution, with tiling and token merging, and what that does to context and KV-cache memory.
- Describe the alignment, pretraining, and instruction-tuning stages, and compare MLP projectors with Q-Former and cross-attention designs.
- Evaluate and monitor visual hallucination, and decide when a VLM is the wrong tool.

**Why it matters.** [CLIP](05-clip-contrastive-pretraining.md) can say which caption fits an image best, but it cannot write an answer. VLMs can: read a chart, extract fields from an invoice, describe a defect, answer "is the left door open?", or look at a screenshot and decide where to click. Hosted frontier models and open families (Qwen-VL, InternVL, LLaVA-style, Gemma) all use variations of the same recipe, so once you know it you can reason about cost, failure modes, and fine-tuning for any of them.

## 1. Intuition

**Three parts.**

1. **Vision encoder.** Usually a [ViT](03-vision-transformers.md) pretrained with CLIP or SigLIP, sometimes a self-supervised one. It turns the image into a grid of patch features, for example 576 vectors of width 1024 for a 336-pixel image with 14-pixel patches.
2. **Projector.** A small network, often a two-layer MLP, that maps each patch feature from the vision width to the LLM's embedding width (say 1024 to 4096). This is the translator. It is cheap to train and carries most of the cross-modal alignment.
3. **Language model.** A normal decoder-only LLM. It sees one sequence: text token embeddings with the projected image vectors spliced in where the image belongs. It generates text as usual.

Why does a CLIP-style encoder make this easy? Its features were already trained to line up with language, so a small projector is enough to reach the LLM's word space. A vision encoder trained only on ImageNet labels would need a much bigger bridge.

**Splicing.** The prompt template contains a placeholder such as `<image>`. The processor expands it into exactly as many placeholder tokens as the image will produce. Before the LLM runs, those positions in the embedding sequence are overwritten with the projected image vectors. From then on, attention treats them like any other tokens: text can attend to image patches and vice versa.

**Training happens in stages.**

1. **Alignment.** Freeze the vision encoder and the LLM. Train only the projector on image-caption pairs, so projected image vectors land where the LLM expects related words.
2. **Pretraining or joint training.** Unfreeze more (the LLM, sometimes the vision encoder) and train on large amounts of image-text data: captions, interleaved web documents, OCR, charts.
3. **Instruction tuning.** Fine-tune on (image, question, answer) conversations so the model follows instructions and answers in the expected format. Preference tuning may follow.

Most teams who adapt a VLM only do a version of stage 3 with LoRA on their own examples ([fine-tuning in practice](../15-llms/04-fine-tuning-in-practice.md)).

**Other bridges.** BLIP-2's *Q-Former* uses a fixed number of learned queries (32) that cross-attend to the image features, so every image costs 32 tokens regardless of resolution. Flamingo-style models keep image features out of the sequence entirely and add *gated cross-attention* layers inside the LLM. The MLP projector won in most open models because it is simple and keeps every patch, at the price of more tokens.

**Image tokens are expensive.** Each image adds hundreds or thousands of tokens to the context. Higher resolution helps reading small text but multiplies the cost. Two common tricks: **tiling** (cut a large image into encoder-sized tiles plus a low-resolution thumbnail) and **token merging** (combine each 2 by 2 block of neighbouring patch features into one token before the projector, cutting tokens by 4).

## 2. Visualization

![Left: bar chart of image tokens per picture: 256 at 224 pixels with 14-pixel patches, 576 at 336 pixels, 144 after 2 by 2 token merging, and 2880 for a 672-pixel image tiled into four 336-pixel tiles plus a thumbnail. Right: nearest-word accuracy of projected synthetic image features rising from 32 percent at step 0 to 100 percent as only the projector is trained.](../../figures/vision-language-models.png)

*Left: what one image costs in LLM context. Right: stage-1 alignment on synthetic data. With the vision features and the word embeddings frozen, training only the projector teaches it to map each image class next to the right word.*

**Try it** (predict first, then check with the script in section 4):
1. A 448-pixel image with 14-pixel patches: how many tokens? With 2 by 2 merging?
2. A prompt has three `<image>` placeholders but the processor produced features for two images. Predict what the merge function should do, and why silently continuing is dangerous.
3. Halve the projector's hidden width from 64 to 32 in the alignment run. Predict whether it still reaches 100 percent on this easy synthetic task.
4. A user uploads ten 672-pixel screenshots to a model with a 32k context using tiling. How much room is left for text?

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $H, P$ | image side and patch side in pixels | scalars |
| $N_v$ | image tokens, $(H/P)^2$ per tile | scalar |
| $V$ | vision encoder output | $[N_v, d_v]$ |
| $W_1, W_2$ | projector weights | $[d_v, d_h]$, $[d_h, d_\ell]$ |
| $\tilde{V}$ | projected image tokens | $[N_v, d_\ell]$ |
| $E$ | text token embeddings | $[M, d_\ell]$ |
| $X$ | merged input sequence | $[M, d_\ell]$ |

### Projector and merge

```math
\tilde{V} = \text{GELU}(V W_1 + b_1)\, W_2 + b_2
```

```math
X_m = \begin{cases} \tilde{V}_{k(m)} & \text{if token } m \text{ is the } k\text{-th } \texttt{<image>} \text{ placeholder} \\ E_m & \text{otherwise} \end{cases}
```

The LLM then runs on $X$ exactly as it would on text. The training loss is next-token cross-entropy on the *answer* tokens only; image positions and the prompt are masked out of the loss.

### Token budget

```math
N_v = \left(\frac{H}{P}\right)^2, \qquad N_v^{\text{merged}} = \frac{N_v}{r^2}, \qquad N_v^{\text{tiled}} = (T + 1)\, N_v
```

for an $r \times r$ merge and $T$ tiles plus one thumbnail.

### Worked example 1: a tiny projector by hand

One image token $v = (1, 2)$, $d_v = 2$. A linear projector (no hidden layer, for clarity) into an LLM with $d_\ell = 3$:

```math
W = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \end{pmatrix}, \quad b = (0.5, 0, 0)
```

1. $vW = (1 \cdot 1 + 2 \cdot 0,\; 1 \cdot 0 + 2 \cdot 1,\; 1 \cdot 2 + 2 \cdot (-1)) = (1, 2, 0)$.
2. Add the bias: $\tilde{v} = (1.5, 2, 0)$.
3. The prompt is `[<s>, USER:, <image>, what, ?]` with embeddings $E$ of shape $[5, 3]$. Row 2 is replaced by $(1.5, 2, 0)$. The LLM receives a $[5, 3]$ sequence and never sees pixels.

Real projectors do the same with $d_v = 1024$, a hidden width of 4096, $d_\ell = 4096$, and GELU in the middle.

### Worked example 2: projector size

A two-layer MLP from 1024 to 4096 to 4096:

```math
1024 \times 4096 + 4096 + 4096 \times 4096 + 4096 = 4{,}194{,}304 + 4{,}096 + 16{,}777{,}216 + 4{,}096 = 20{,}979{,}712
```

About 21 million parameters, roughly 0.3 percent of a 7-billion-parameter LLM. That is why stage-1 alignment, training only this piece, is cheap.

### Worked example 3: tokens and KV-cache memory

Take a ViT with 14-pixel patches at 336 pixels and a 7B-class LLM with 32 layers, 32 KV heads of width 128, in fp16.

1. Tokens for one 336-pixel image: $(336/14)^2 = 24^2 = 576$.
2. A 672-pixel image tiled into four 336-pixel tiles plus a thumbnail: $5 \times 576 = 2880$ tokens.
3. With 2 by 2 merging: $2880 / 4 = 720$ tokens.
4. KV cache per token: $2 \times 32 \text{ layers} \times 32 \text{ heads} \times 128 \times 2 \text{ bytes} = 524{,}288$ bytes $= 0.5$ MiB (the same arithmetic as [self-attention](../14-transformers/01-self-attention.md)).
5. One tiled image: $2880 \times 0.5 = 1440$ MiB, about 1.4 GiB per sequence before any text. Merged: $720 \times 0.5 = 360$ MiB.

Ten such images unmerged would be 28,800 tokens, nearly filling a 32k context. Image resolution is a cost decision, not just a quality one.

## 4. Implementation

The full script is [`code/25-computer-vision/vlm_scratch.py`](../../code/25-computer-vision/vlm_scratch.py) (NumPy only). It counts tokens and projector parameters, splices image tokens into a prompt, checks for placeholder mismatches, and runs a stage-1 alignment in which only the projector learns.

### Build it from scratch

**Step 1: the projector.** Two linear layers with GELU between them. Each image token is translated independently; there is no mixing across patches here (the LLM's attention does that later).

```python
def project(v, P):
    """v: [N_img, d_vit] patch features -> [N_img, d_llm] 'soft tokens' the LLM reads like words."""
    return gelu(v @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
```

*Predict:* if you removed the GELU, how many parameters could you save by multiplying $W_1 W_2$ into one matrix, and what would you lose?

**Step 2: splice image tokens into the sequence.** Look up the ordinary word embeddings, find the placeholder positions, overwrite them. The count check matters: a mismatch means the prompt template and the image processor disagree, and silently truncating would shift every later token.

```python
def merge(text_ids, embed_table, image_tokens, image_id):
    """Replace each <image> placeholder id with the next projected image token."""
    out = embed_table[text_ids].copy()                         # [M, d_llm] ordinary word embeddings
    slots = np.flatnonzero(text_ids == image_id)
    if len(slots) != len(image_tokens):                        # most common VLM porting bug: say so loudly
        raise ValueError(f"{len(slots)} <image> placeholders but {len(image_tokens)} image tokens")
    out[slots] = image_tokens
    return out
```

*Predict:* the image tokens sit at positions 2 to 5 in the sequence. Under a causal mask, can the image tokens attend to the question that comes after them? Does it matter for answering?

**Step 3: stage-1 alignment, only the projector learns.** The "vision encoder" is a frozen set of class prototypes plus noise, and the "LLM vocabulary" is a frozen set of word embeddings. The projector is trained so each projected image lands on its class word, using mean squared error and hand-written backprop through the two layers.

```python
    for step in range(1, steps + 1):                           # MSE to the caption word embedding, plain GD
        h_pre = feats @ P["W1"] + P["b1"]
        h = gelu(h_pre)
        err = (h @ P["W2"] + P["b2"]) - words[y]                # [n, d_llm]
        g_out = 2 * err / len(y)
        dW2, db2 = h.T @ g_out, g_out.sum(0)
        t = np.tanh(np.sqrt(2 / np.pi) * (h_pre + 0.044715 * h_pre ** 3))
        dgelu = 0.5 * (1 + t) + 0.5 * h_pre * (1 - t ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * h_pre ** 2)
        g_h = (g_out @ P["W2"].T) * dgelu
        dW1, db1 = feats.T @ g_h, g_h.sum(0)
        for k, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
            P[k] -= lr * g
```

Real stage-1 training uses the LLM's own next-token loss on captions instead of MSE to a word vector, but the frozen-everything-except-the-bridge structure is the same.

*Predict:* accuracy starts at 32 percent, above the 20 percent chance level for 5 classes. Why can a random projector already beat chance?

### Use the library

Hosted APIs accept images directly. Open models run through Hugging Face `transformers` (or `vllm` for serving); the processor builds the placeholders and the model does the merge.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor   # pip install transformers torch pillow

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"                              # any chat VLM checkpoint
proc = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
messages = [{"role": "user", "content": [{"type": "image", "image": pil_image},
                                         {"type": "text", "text": "List every line item and its price as JSON."}]}]
inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300)
print(proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

What the library adds: the image preprocessing and tiling rules the checkpoint was trained with, the exact number of placeholder tokens, the chat template, and the merge inside the forward pass. Many processors expose resolution or pixel-budget settings; use them to control token cost.

### Run it

```bash
uv run code/25-computer-vision/vlm_scratch.py
```

Key lines of output (synthetic features):

```text
ViT-L/14 at 336: (336/14)^2 = 576 image tokens
params = 1024*4096 + 4096 + 4096*4096 + 4096 = 20,979,712
text ids [1, 7, 99, 99, 99, 99, 23, 42, 5] -> sequence (9, 12); rows 2..5 are projected image tokens
mismatch caught: 3 <image> placeholders but 4 image tokens
672x672 as 4 tiles of 336 + 1 thumbnail: 5 x 576 = 2880 tokens
step   0: nearest-word accuracy 32%
step  50: nearest-word accuracy 98%
step 300: nearest-word accuracy 100%
```

## 5. Engineering

**Use cases.** Document and form extraction (invoices, receipts, IDs), chart and table question answering, visual inspection reports, accessibility (alt text, scene description), retail catalogue enrichment, screenshot understanding for UI agents ([tool use and agents](../19-agents/01-tool-use-and-agents.md)), and video question answering by sampling frames.

**Hosted, open, or fine-tuned.** A hosted API is the fastest start and usually the strongest general model, but check data handling and per-image pricing. An open model self-hosted gives control over data and cost at volume. Fine-tune an open model with LoRA when you need a fixed output format, domain vocabulary, or behaviour the base model lacks; a few thousand good examples often go a long way. Evaluate all three on *your* images before deciding.

**Control the token budget.** Pick the lowest resolution that preserves what matters (small text needs more), crop to the region of interest, cap the number of images per request, and prefer models with token merging for multi-image or video inputs. Image tokens are prefilled like text, so they add to time-to-first-token and to KV-cache memory ([LLM inference](../18-llm-inference/01-llm-inference.md)).

**Do not use a VLM where a specialist is better.** For counting many small objects, precise boxes, measurements, or pixel masks, a detector or segmenter ([object detection](01-object-detection-yolo.md), [segmentation](02-semantic-segmentation-unet.md)) is more accurate, cheaper, and easier to evaluate. A common pattern is specialist model for geometry, VLM for the language around it.

**Evaluate grounding, not fluency.** Build a labelled set from your own images: exact-match field extraction, yes/no questions about present and absent objects (the POPE-style probe for object hallucination), counting, and spatial questions. Track the error rate on *absent-object* questions separately; that is where confident hallucination shows up.

**Monitor hallucination in production.** Useful signals: an independent CLIP or SigLIP similarity between the image and the claims in the answer (flag answers that are confident but poorly grounded for review), agreement between two prompts or two models, schema validation for extraction, and periodic human audits stratified by input type. Treat these as triage signals, not proof.

> [!WARNING]
> **Failure modes.** Describing objects that are not there (strong language priors: "a kitchen" implies "a fridge"); misreading small or rotated text at low resolution; weak counting and left-right reasoning; ignoring the image and answering from the question alone; following instructions written *inside* the image (a visual [prompt injection](../../glossary.md#prompt-injection)); and placeholder-count or preprocessing mismatches when porting a model.

### Common mistakes

- Judging a VLM by how fluent its descriptions sound instead of by grounded, labelled questions.
- Sending full-resolution images by default and paying thousands of tokens per image.
- Fine-tuning the vision encoder with a high learning rate and destroying its pretrained features (freeze it or use a much smaller rate).
- Computing the loss on prompt and image positions instead of answer tokens only.
- Trusting text extracted from images without validation, or letting text inside an image act as instructions.
- Using a VLM to count or measure when a detector would do it reliably.

## 6. Knowledge check

<!-- quiz:vision-language-models -->
**[Take the vision-language models quiz](../../quizzes/vision-language-models.md)**: architecture, token budgets, training stages, and hallucination.
<!-- /quiz -->

**Practice exercise.** A VLM uses a ViT with 16-pixel patches at 384 pixels, an MLP projector from 1152 to 3584 to 3584, and no token merging. How many image tokens does one image produce? How many projector parameters are there? If you add 2 by 2 merging (concatenating four neighbours into one 4608-wide vector before the projector), what are the new token count and projector size?

<details>
<summary>Solution</summary>

Tokens: $(384/16)^2 = 24^2 = 576$. Projector: $1152 \times 3584 + 3584 + 3584 \times 3584 + 3584 = 4{,}128{,}768 + 3{,}584 + 12{,}845{,}056 + 3{,}584 = 16{,}980{,}992$, about 17.0M. With 2 by 2 merging: $576/4 = 144$ tokens, and the first layer now takes $4 \times 1152 = 4608$ inputs: $4608 \times 3584 + 3584 + 12{,}845{,}056 + 3584 = 16{,}515{,}072 + 3{,}584 + 12{,}845{,}056 + 3{,}584 = 29{,}367{,}296$, about 29.4M. Four times fewer tokens for about 12M more projector parameters, a good trade because tokens are paid on every request.

</details>

**Implementation challenge.** Add 2 by 2 token merging to the script: take a $[16, d_v]$ grid of patch features from a 4 by 4 grid, concatenate each 2 by 2 block into one $[4 d_v]$ vector (giving $[4, 4 d_v]$), widen the projector's first layer to match, and check the merged prompt needs only 4 placeholders.

<details>
<summary>Solution sketch</summary>

```python
def merge_2x2(v, g):                                   # v: [g*g, d] in row-major grid order
    d = v.shape[1]
    x = v.reshape(g // 2, 2, g // 2, 2, d).transpose(0, 2, 1, 3, 4)   # [g/2, g/2, 2, 2, d]
    return x.reshape((g // 2) ** 2, 4 * d)

vision16 = rng.standard_normal((16, d_vit))
merged = merge_2x2(vision16, 4)                        # [4, 4 * d_vit]
P4 = {"W1": rng.standard_normal((4 * d_vit, d_llm)), "b1": np.zeros(d_llm),
      "W2": rng.standard_normal((d_llm, d_llm)), "b2": np.zeros(d_llm)}
seq = merge(np.array([1, 7, IMG, IMG, IMG, IMG, 23]), table, project(merged, P4), IMG)
```

The reshape and transpose are the same trick as ViT patchify, applied to feature vectors instead of pixels. The sequence now holds 4 image tokens, each carrying four patches' worth of information.

</details>

## Summary

- A VLM is a vision encoder (usually a CLIP or SigLIP ViT), a projector (often a two-layer MLP), and a decoder LLM; projected image features replace `<image>` placeholders in the token sequence.
- Training goes alignment (projector only), joint pretraining, then instruction tuning; most adaptation is LoRA on instruction data.
- One image costs $(H/P)^2$ tokens per tile; tiling multiplies it and token merging divides it. Image tokens consume context, prefill time, and KV-cache memory.
- Q-Former and cross-attention are alternative bridges that trade detail for fewer tokens.
- Evaluate grounding on labelled questions, especially absent objects, monitor hallucination with independent signals, and use specialist models for geometry.

**Next:** [Spectrograms and mel features](../29-speech-and-audio/01-spectrograms-mel-features.md)

**Related:** [CLIP](05-clip-contrastive-pretraining.md) · [Vision transformers](03-vision-transformers.md) · [Fine-tuning in practice](../15-llms/04-fine-tuning-in-practice.md) · [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md) · [Tool use and agents](../19-agents/01-tool-use-and-agents.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [OCR & Document Understanding](https://aiengineeringfromscratch.com/lesson?path=phases/04-computer-vision/19-ocr-document-understanding) · [From CLIP to BLIP-2 — Q-Former as Modality Bridge](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/03-blip2-qformer-bridge) · [Flamingo and Gated Cross-Attention for Few-Shot VLMs](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/04-flamingo-gated-cross-attention) · [LLaVA and Visual Instruction Tuning](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/05-llava-visual-instruction-tuning) · [Open-Weight VLM Recipes: What Actually Matters](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/07-open-weight-vlm-recipes)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Vision-Language Models — The ViT-MLP-LLM Pattern](https://aiengineeringfromscratch.com/lesson?path=phases/04-computer-vision/25-vision-language-models).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>How does a modern open VLM turn an image and a question into an answer?</strong></summary>

The image is resized or tiled to the encoder's resolution and split into patches. A pretrained vision transformer, usually CLIP- or SigLIP-trained, turns it into patch features: 576 vectors of width 1024 for a 336-pixel image with 14-pixel patches. A projector, typically a two-layer MLP, maps each feature into the LLM's embedding width. The chat template contains an image placeholder that the processor expands to exactly 576 placeholder tokens. Before the decoder runs, those positions in the embedding sequence are overwritten with the projected vectors. The LLM then processes one sequence of text and image tokens with ordinary causal attention and generates the answer token by token. Nothing in the decoder is image-specific. The CLIP-style encoder matters because its features are already aligned with language, so a small projector is enough to bridge the two.

</details>

<details>
<summary><strong>MLP projector versus Q-Former versus cross-attention: what are the trade-offs?</strong></summary>

An MLP projector keeps one token per patch, hundreds or thousands per image. It preserves fine detail such as small text and is simple to train, which is why most open VLMs use it. The cost is context length and prefill compute that grow with resolution. A Q-Former, as in BLIP-2, uses a fixed set of learned queries, say 32, that cross-attend to the patch features, so every image costs the same few tokens. That is efficient for many images or video, but it can lose detail needed for OCR or dense documents. Flamingo-style gated cross-attention keeps image features outside the sequence and adds cross-attention layers inside the LLM. Image cost does not touch the context window, but the LLM's architecture changes and the new layers need substantial training. Token merging and tiling let MLP designs move along the detail-versus-cost curve.

</details>

<details>
<summary><strong>Users report that your VLM confidently describes objects that are not in the photo. How do you measure and reduce it?</strong></summary>

First measure it properly. Build a labelled set from production-like images with yes/no questions about present and absent objects, the POPE-style probe, plus your real task questions. Report the false-yes rate on absent objects separately, since aggregate accuracy hides it. Check whether failures correlate with low resolution, cluttered scenes, or leading prompts like "describe the fridge". Reductions: raise resolution or crop to the region of interest; reword prompts so they don't presuppose objects, and allow "not visible" as an answer; constrain outputs with a schema; and fine-tune with examples that include absent-object questions answered "no". In production, add independent grounding checks, such as a CLIP or SigLIP similarity between the image and each claimed object, or a detector for objects that matter, and route low-grounding, high-confidence answers to review.

</details>

<details>
<summary><strong>Quick calculation: how much KV-cache memory does one tiled 672-pixel image take in a 7B-class VLM?</strong></summary>

Count tokens first. With 14-pixel patches at 336 pixels each tile is 24 × 24 = 576 tokens. Four tiles plus a thumbnail is 5 × 576 = 2880 tokens. For a 7B-class model with 32 layers, 32 KV heads of width 128, and fp16, each token stores keys and values: 2 × 32 × 32 × 128 × 2 bytes = 524,288 bytes, half a mebibyte. So 2880 × 0.5 MiB = 1440 MiB, about 1.4 GiB per sequence for the image alone, before any text. Ten such images would be 28,800 tokens and about 14 GiB. A 2 by 2 token merge cuts that by four, to about 360 MiB per image, and grouped-query attention with 8 KV heads cuts it by another four. Resolution policy is therefore a serving-cost lever.

</details>
