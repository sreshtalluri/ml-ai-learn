<!-- GENERATED from autoencoders-diffusion-and-transfer.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Autoencoders, diffusion, and transfer learning

Covers the lesson [Autoencoders, diffusion, and transfer learning](../lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/autoencoders-diffusion-and-transfer/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

With $\bar\alpha_t = 0.81$, $x_0 = 3$, and $\epsilon = 1$, what is $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$? (3 decimals)

<details>
<summary>Answer</summary>

**3.136** (within ±0.002)

$0.9 \times 3 + \sqrt{0.19} \times 1 = 2.7 + 0.436 = 3.136$.

</details>

## 2. Calculation (easy)

A frozen backbone outputs 768-dimensional features. How many trainable parameters does a new 3-class linear head (with biases) have?

<details>
<summary>Answer</summary>

**2307**

$768 \times 3 + 3 = 2307$.

</details>

## 3. Multiple choice (easy)

Why does an autoencoder need a bottleneck (or another constraint)?

- **A.** To make training faster.
- **B.** Without it, the network could learn the identity function and copy inputs without learning useful structure.
- **C.** Because decoders can't be wide.
- **D.** To make outputs probabilities.

<details>
<summary>Answer</summary>

**B.** Without it, the network could learn the identity function and copy inputs without learning useful structure.

The constraint forces a compressed representation.

</details>

## 4. Multiple choice (medium)

What does a standard diffusion model's network learn to predict during training?

- **A.** The class label of the image
- **B.** The noise that was added to a clean sample at a random step
- **C.** The next pixel
- **D.** The number of diffusion steps

<details>
<summary>Answer</summary>

**B.** The noise that was added to a clean sample at a random step

The simplified DDPM loss is the squared error between the true and predicted noise.

</details>

## 5. Match (medium)

Match the situation to a transfer-learning strategy.

| Concept | Options |
|---|---|
| 300 labeled images, similar to the pretraining domain | Fine-tune everything or pretrain in-domain |
| 20,000 labeled images | Fine-tune top layers with a small learning rate |
| Millions of in-domain images, very different domain | Freeze the backbone and train a new head |

<details>
<summary>Answer</summary>

- 300 labeled images, similar to the pretraining domain → Freeze the backbone and train a new head
- 20,000 labeled images → Fine-tune top layers with a small learning rate
- Millions of in-domain images, very different domain → Fine-tune everything or pretrain in-domain

More data and a bigger domain gap justify updating more of the model.

</details>

## 6. Reflection (hard)

Describe how to build an autoencoder-based anomaly detector for server metrics and one way it can fail.

<details>
<summary>Answer</summary>

**Model answer.** Train an autoencoder on windows of metrics from normal operation only, compute reconstruction error on a validation set of normal windows, and set the alert threshold at a high percentile (tuned with any labeled incidents). Flag new windows whose error exceeds it. It can fail if the autoencoder generalizes too well and reconstructs anomalies accurately, if "normal" training data contains unlabeled incidents, or if normal behavior drifts (new traffic patterns) and creates false alarms.

Reconstruction error ranks unusualness; thresholds and training data quality decide usefulness.

</details>
