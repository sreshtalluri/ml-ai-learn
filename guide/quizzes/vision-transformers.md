<!-- GENERATED from vision-transformers.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Vision transformers

Covers the lesson [Vision transformers](../lessons/25-computer-vision/03-vision-transformers.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/vision-transformers/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

How does a ViT turn an image into a sequence of tokens?

- **A.** A convolution with kernel size equal to stride equal to the patch size, which splits the image into patches and projects each one in a single operation
- **B.** Flattening the whole image into one vector
- **C.** Running a CNN and using its final feature map as tokens
- **D.** Sorting pixels by intensity

<details>
<summary>Answer</summary>

**A.** A convolution with kernel size equal to stride equal to the patch size, which splits the image into patches and projects each one in a single operation

At 224 pixels with 16-pixel patches the convolution produces a 14 by 14 grid of D-dimensional vectors, flattened to 196 tokens.

- **A:** Correct.
- **B:** One vector would be a single token; attention would have nothing to attend over.
- **C:** That is a hybrid design some models use, but the standard ViT patchifies the raw image.
- **D:** Sorting would destroy spatial structure.

</details>

## 2. Multiple choice (easy)

What is the [CLS] token in a ViT?

- **A.** A placeholder for missing patches
- **B.** A special pixel value
- **C.** A dropout mask
- **D.** A learned vector prepended to the patch tokens; its final output summarises the image and feeds the classifier

<details>
<summary>Answer</summary>

**D.** A learned vector prepended to the patch tokens; its final output summarises the image and feeds the classifier

Attention lets the [CLS] token gather information from every patch. Many later models use mean pooling over patch tokens instead.

- **A:** Missing patches are an MAE concept, handled differently.
- **B:** It is a learned embedding, not a pixel.
- **C:** It has nothing to do with dropout.
- **D:** Correct.

</details>

## 3. Calculation (easy)

A ViT uses 14-pixel patches on a 448 by 448 image and a [CLS] token. How many tokens enter the first encoder block?

<details>
<summary>Answer</summary>

**1025**

$448/14 = 32$ patches per side, $32^2 = 1024$ patches, plus [CLS] gives 1025.

</details>

## 4. Multiple choice (medium)

The original ViT lost to ResNets when trained only on ImageNet-1k, yet DeiT matched them on the same data. What changed?

- **A.** DeiT used a different attention mechanism
- **B.** The original paper had a bug
- **C.** ViTs lack CNN priors like locality, so they overfit with little data; DeiT compensated with strong augmentation, regularisation, and distillation
- **D.** ViTs cannot work without 300 million images

<details>
<summary>Answer</summary>

**C.** ViTs lack CNN priors like locality, so they overfit with little data; DeiT compensated with strong augmentation, regularisation, and distillation

Weaker built-in assumptions mean the network must learn them from data. DeiT's recipe (RandAugment, Mixup, CutMix, stochastic depth, distillation) supplies that.

- **A:** DeiT kept standard attention.
- **B:** The result was real; it reflected the training recipe and data scale.
- **C:** Correct.
- **D:** DeiT showed that is false.

</details>

## 5. Multiple choice (medium)

Why do ViTs and modern LLMs use pre-LayerNorm, $x + f(\text{LN}(x))$, rather than post-LayerNorm, $\text{LN}(x + f(x))$?

- **A.** It uses fewer parameters
- **B.** The residual stream is never normalised, so deep stacks train stably without delicate warmup
- **C.** They are mathematically identical
- **D.** Post-LN cannot be implemented on GPUs

<details>
<summary>Answer</summary>

**B.** The residual stream is never normalised, so deep stacks train stably without delicate warmup

With post-LN every block's output passes through a normalization, which makes gradients harder to control as depth grows. Pre-LN keeps a clean residual path.

- **A:** Both use the same LayerNorm parameters.
- **B:** Correct.
- **C:** They behave differently in training.
- **D:** Both are easy to implement.

</details>

## 6. Multiple choice (medium)

What problem does Swin Transformer's windowed attention solve?

- **A.** Missing batch norm support
- **B.** Overfitting to texture
- **C.** Vanishing gradients
- **D.** Full attention cost grows with the square of the patch count; attending within local windows makes it grow linearly, and shifted windows let information cross windows

<details>
<summary>Answer</summary>

**D.** Full attention cost grows with the square of the patch count; attending within local windows makes it grow linearly, and shifted windows let information cross windows

At high resolution with small patches there are thousands of tokens. Local windows make attention cost proportional to the number of tokens times the window size.

- **A:** Unrelated.
- **B:** Unrelated.
- **C:** Residual connections handle gradients.
- **D:** Correct.

</details>

## 7. Calculation (hard)

Using the approximation of $12D^2$ parameters per encoder block, roughly how many million parameters are in the 24 blocks of ViT-L ($D = 1024$)? Round to the nearest million.

<details>
<summary>Answer</summary>

**302** (within ±2)

$12 \times 1024^2 = 12{,}582{,}912$ per block, times 24 blocks $= 301{,}989{,}888 \approx 302$ million. The full ViT-L/16 is about 304 million with embeddings and head.

</details>

## 8. Multiple choice (hard)

You load a ViT pretrained at 224 pixels and fine-tune at 384 pixels. Loading fails with a shape mismatch on `pos_embed` (197 versus 577). What is the right fix?

- **A.** Truncate the new sequence to 197 tokens
- **B.** Reinitialize the position embeddings randomly
- **C.** Reshape the 196 patch positions into a 14 by 14 grid, interpolate in 2-D to 24 by 24, flatten, and keep the [CLS] position as is
- **D.** Switch to 32-pixel patches

<details>
<summary>Answer</summary>

**C.** Reshape the 196 patch positions into a 14 by 14 grid, interpolate in 2-D to 24 by 24, flatten, and keep the [CLS] position as is

2-D interpolation preserves the learned spatial layout. Libraries such as timm do this when you set a new image size.

- **A:** Truncation drops most of the image.
- **B:** Random positions throw away learned structure and slow fine-tuning.
- **C:** Correct.
- **D:** Changing the patch size breaks the pretrained patch-embedding weights.

</details>

## 9. Multiple choice (medium)

You need general-purpose frozen image features for retrieval across a product catalogue, with no labels. Which backbone is the best starting point?

- **A.** A ViT trained from scratch on your catalogue with a classification head
- **B.** A self-supervised or contrastive pretrained ViT (DINOv2- or CLIP/SigLIP-style), used frozen
- **C.** An ImageNet-supervised ResNet's logits
- **D.** A U-Net encoder

<details>
<summary>Answer</summary>

**B.** A self-supervised or contrastive pretrained ViT (DINOv2- or CLIP/SigLIP-style), used frozen

Self-supervised and contrastive backbones are trained to produce features that work well frozen for similarity search.

- **A:** You have no labels, and training from scratch needs lots of data.
- **B:** Correct.
- **C:** Class logits are tuned to 1000 ImageNet classes and are weak for fine-grained retrieval.
- **D:** Segmentation encoders are not trained for whole-image similarity.

</details>
