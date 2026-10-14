<!-- GENERATED from semantic-segmentation-unet.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Semantic segmentation and U-Net

Covers the lesson [Semantic segmentation and U-Net](../lessons/25-computer-vision/02-semantic-segmentation-unet.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/semantic-segmentation-unet/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Why does U-Net need skip connections between encoder and decoder?

- **A.** Transposed convolutions require extra inputs
- **B.** To match framework shape conventions
- **C.** The encoder trades spatial detail for context; skips hand the decoder the encoder's high-resolution features so it can place edges precisely
- **D.** To speed up training

<details>
<summary>Answer</summary>

**C.** The encoder trades spatial detail for context; skips hand the decoder the encoder's high-resolution features so it can place edges precisely

Decoding from the bottleneck alone gives blocky edges and loses small objects. Concatenating encoder features at each resolution supplies the detail.

- **A:** Upsampling layers need no extra inputs.
- **B:** Shapes are not the reason; skips change what information is available.
- **C:** Correct.
- **D:** Speed is not the purpose, though shorter gradient paths help training a little.

</details>

## 2. Multiple choice (medium)

A task has 99 percent background and 1 percent tumour pixels. Training with plain cross-entropy reaches 99 percent pixel accuracy. What most likely happened?

- **A.** The loss was implemented incorrectly
- **B.** The model converged to a good solution
- **C.** The model predicts background almost everywhere; pixel accuracy is dominated by the majority class
- **D.** The optimizer failed

<details>
<summary>Answer</summary>

**C.** The model predicts background almost everywhere; pixel accuracy is dominated by the majority class

All-background predictions score 99 percent accuracy and 0 tumour IoU. Add Dice (or focal) loss and report per-class IoU or Dice.

- **A:** A correct cross-entropy produces exactly this behaviour on imbalanced data.
- **B:** 99 percent accuracy is what doing nothing scores here.
- **C:** Correct.
- **D:** The optimizer found a real minimum of the wrong objective.

</details>

## 3. Multiple choice (easy)

Which task gives each of several touching cars its own mask?

- **A.** Semantic segmentation
- **B.** Instance segmentation
- **C.** Image classification
- **D.** Neither; you need a detector

<details>
<summary>Answer</summary>

**B.** Instance segmentation

Semantic segmentation merges touching cars into one "car" region. Instance segmentation (and panoptic) keeps separate object ids.

- **A:** Semantic labels say "car" but not which car.
- **B:** Correct.
- **C:** Classification gives one label per image.
- **D:** Instance segmentation often uses a detector inside it, but its output is per-object masks.

</details>

## 4. Multiple choice (hard)

A decoder using `ConvTranspose2d(kernel_size=3, stride=2)` produces checkerboard patterns. Why?

- **A.** Transposed convolution is broken in PyTorch
- **B.** The learning rate is too low
- **C.** Kernel size is not divisible by stride, so alternate output pixels receive different numbers of contributions
- **D.** Batch norm is missing

<details>
<summary>Answer</summary>

**C.** Kernel size is not divisible by stride, so alternate output pixels receive different numbers of contributions

With kernel 3 and stride 2, interior outputs alternate between 1 and 2 contributions, creating a periodic pattern. Kernel 4 with stride 2, or bilinear upsampling plus a convolution, gives even coverage.

- **A:** It is a property of the operation, not a bug.
- **B:** Learning rate does not create a fixed spatial period.
- **C:** Correct.
- **D:** Normalization does not fix uneven overlap.

</details>

## 5. Multiple choice (medium)

You report mIoU = 0.78 on 10 classes. Why also publish per-class IoU?

- **A.** A mean can hide failures; 0.78 is consistent with eight classes at 0.9 and two at 0.3
- **B.** Frameworks require it
- **C.** Only medical imaging needs it
- **D.** Per-class IoU is always higher than mIoU

<details>
<summary>Answer</summary>

**A.** A mean can hide failures; 0.78 is consistent with eight classes at 0.9 and two at 0.3

Deployment decisions depend on the weakest classes the product needs, which the mean conceals.

- **A:** Correct.
- **B:** No framework requires it.
- **C:** Any multi-class task can hide a weak class.
- **D:** By definition some classes are at or below the mean.

</details>

## 6. Calculation (medium)

A model predicts 60 tumour pixels; the truth has 50; 40 pixels are in both. What is the Dice score for the tumour class? (Three decimals.)

<details>
<summary>Answer</summary>

**0.727** (within ±0.002)

Dice $= 2 \times 40/(60 + 50) = 80/110 = 0.727$. IoU is $40/70 = 0.571$, and $2 \times 0.571/1.571 = 0.727$ confirms it.

</details>

## 7. Fill in (medium)

A four-level U-Net with base width 64 receives a 512 by 512 image. What is the bottleneck's spatial size per side? Answer with a number.

<details>
<summary>Answer</summary>

**32**

Four poolings divide each side by $2^4 = 16$, so $512/16 = 32$. The bottleneck is $[1024, 32, 32]$.

</details>

## 8. Multiple choice (hard)

After resizing images and masks with bilinear interpolation, training masks contain class id 1.5 on borders, and IoU for class 2 is oddly high near class-1 edges. What is the fix?

- **A.** Use a higher learning rate
- **B.** Resize masks with nearest-neighbour interpolation (images can stay bilinear)
- **C.** Add Dice loss
- **D.** Normalize the masks

<details>
<summary>Answer</summary>

**B.** Resize masks with nearest-neighbour interpolation (images can stay bilinear)

Bilinear interpolation averages class ids and creates labels that don't exist (rounding 1.5 can give class 2). Masks must use nearest-neighbour so every pixel keeps a valid id.

- **A:** Optimisation can't fix corrupted labels.
- **B:** Correct.
- **C:** The loss is not the problem; the targets are wrong.
- **D:** Class ids are categories; normalizing them is meaningless.

</details>

## 9. Multiple choice (hard)

You must segment 4000 by 4000 pixel satellite tiles on a GPU with 16 GB. Which plan is most sensible?

- **A.** Downsample to 256 by 256 and upsample the prediction
- **B.** Train on random 512 by 512 crops and predict with overlapping tiles stitched together, using a pretrained encoder
- **C.** Train on full 4000 by 4000 images with batch size 1
- **D.** Use image classification on each tile instead

<details>
<summary>Answer</summary>

**B.** Train on random 512 by 512 crops and predict with overlapping tiles stitched together, using a pretrained encoder

Crops fit memory and keep native resolution; overlapping tiles avoid seams at prediction time; a pretrained encoder reduces label needs.

- **A:** Small buildings and roads would vanish at that downsampling.
- **B:** Correct.
- **C:** Activations at that size do not fit in 16 GB for a U-Net.
- **D:** Classification gives one label per tile, not boundaries.

</details>
