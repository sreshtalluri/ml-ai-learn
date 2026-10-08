<!-- GENERATED from convolutional-networks.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Convolutional networks

Covers the lesson [Convolutional networks](../lessons/13-deep-architectures/01-convolutional-networks.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/convolutional-networks/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A 2×2 kernel $\begin{bmatrix}1 & 0\\0 & -1\end{bmatrix}$ is applied to the patch $\begin{bmatrix}5 & 2\\3 & 1\end{bmatrix}$ (no bias). What is the output value?

<details>
<summary>Answer</summary>

**4**

$5(1) + 2(0) + 3(0) + 1(-1) = 4$.

</details>

## 2. Calculation (medium)

Input 64×64, kernel 3×3, stride 2, padding 1. What is the output height?

<details>
<summary>Answer</summary>

**32**

$\lfloor (64 - 3 + 2)/2 \rfloor + 1 = \lfloor 31.5 \rfloor + 1 = 32$.

</details>

## 3. Calculation (medium)

A conv layer has 3 input channels, 16 output channels, 5×5 kernels, and biases. How many parameters?

<details>
<summary>Answer</summary>

**1216**

$16 \times (3 \times 25 + 1) = 16 \times 76 = 1216$.

</details>

## 4. Fill in (easy)

Max-pool the 2×2 window `[[2, 7], [5, 1]]`. What is the result?

<details>
<summary>Answer</summary>

**7**

Max pooling keeps the largest value.

</details>

## 5. Select all that apply (medium)

Which inductive biases make CNNs data-efficient on images? Select all that apply.

- **A.** Local receptive fields (each unit sees a small patch)
- **B.** Weight sharing (the same kernel at every position)
- **C.** Global attention between every pair of pixels
- **D.** Pooling for tolerance to small shifts

<details>
<summary>Answer</summary>

**A, B, D**

Global attention is the transformer's inductive bias, which needs more data.

</details>

## 6. Reflection (medium)

You have 2,000 labeled product photos in 10 classes. Outline a sensible first approach.

<details>
<summary>Answer</summary>

**Model answer.** Start from a CNN (or vision transformer) pretrained on a large dataset such as ImageNet. Replace the final classification layer with a 10-class head, freeze the backbone at first and train the head, then optionally unfreeze the top layers with a small learning rate. Use label-preserving augmentation and a held-out validation set. Training from scratch on 2,000 images would badly overfit.

Transfer learning plus augmentation is the default for small image datasets.

</details>
