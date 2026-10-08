<!-- GENERATED from neural-network-forward-pass.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: The neural-network forward pass

Covers the lesson [The neural-network forward pass](../lessons/10-neural-networks/01-neural-network-forward-pass.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/neural-network-forward-pass/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

A ReLU neuron has weights $w = [2, -1, 0.5]$ and bias $b = -1$. What is its output for $x = [1, 3, 4]$?

<details>
<summary>Answer</summary>

**0** (within ±0.0001)

$z = 2(1) + (-1)(3) + 0.5(4) - 1 = 2 - 3 + 2 - 1 = 0$. $\text{ReLU}(0) = 0$.

</details>

## 2. Calculation (medium)

Hidden activations are $a = [1, 0, 2]$, output weights $[0.5, 3, -0.25]$, output bias $0$, and the output uses a sigmoid. What is $\hat{y}$? (3 decimals)

<details>
<summary>Answer</summary>

**0.5** (within ±0.001)

$z = 0.5(1) + 3(0) - 0.25(2) + 0 = 0.5 - 0.5 = 0$, and $\sigma(0) = 0.5$. The large weight 3 had no effect because its input activation was 0.

</details>

## 3. Calculation (medium)

How many trainable parameters does a network with layers 10 → 32 → 32 → 3 (all fully connected, with biases) have?

<details>
<summary>Answer</summary>

**1507**

$10 \times 32 + 32 = 352$; $32 \times 32 + 32 = 1056$; $32 \times 3 + 3 = 99$. Total $352 + 1056 + 99 = 1507$.

</details>

## 4. Fill in (medium)

A batch $X$ has shape `[64, 20]`. It passes through `nn.Linear(20, 128)` then `nn.ReLU()` then `nn.Linear(128, 5)`. What is the output shape? Answer like `[a, b]`.

<details>
<summary>Answer</summary>

**[64, 5]** or **[64,5]** or **64,5** or **(64, 5)** or **(64,5)**

The batch dimension is preserved: $[64,20] \to [64,128] \to [64,128] \to [64,5]$.

</details>

## 5. Multiple choice (medium)

Someone builds a 10-layer network but forgets every activation function. On XOR data it reaches about 50% accuracy. Why?

- **A.** Ten layers are too few for XOR.
- **B.** Without nonlinearities the whole network is equivalent to a single linear layer, which cannot separate XOR.
- **C.** The learning rate must be too high.
- **D.** XOR requires convolutional layers.

<details>
<summary>Answer</summary>

**B.** Without nonlinearities the whole network is equivalent to a single linear layer, which cannot separate XOR.

$W_{10}\cdots W_1 x + (\text{bias terms})$ is one matrix times $x$ plus one bias. A linear decision boundary cannot solve XOR, no matter how many layers produced it.

- **A:** One hidden layer with a nonlinearity is enough.
- **B:** Correct.
- **C:** Even perfectly optimized, a linear model cannot exceed chance on XOR.
- **D:** An MLP with a nonlinearity solves it easily.

</details>

## 6. Match (easy)

Match the task to the right output layer.

| Concept | Options |
|---|---|
| Predict a house price | One sigmoid unit per tag, binary cross-entropy each |
| Spam or not spam | 10 units, softmax, categorical cross-entropy |
| Pick one of 10 digit classes | 1 unit, sigmoid, binary cross-entropy |
| Several independent tags per image | 1 unit, no activation, MSE loss |

<details>
<summary>Answer</summary>

- Predict a house price → 1 unit, no activation, MSE loss
- Spam or not spam → 1 unit, sigmoid, binary cross-entropy
- Pick one of 10 digit classes → 10 units, softmax, categorical cross-entropy
- Several independent tags per image → One sigmoid unit per tag, binary cross-entropy each

Softmax forces probabilities to compete and sum to 1, which suits exactly one correct class. Independent tags need independent sigmoids.

</details>

## 7. Select all that apply (hard)

After a few training steps with a large learning rate, 70% of a layer's ReLU units output 0 for every training example. Which statements are true? Select all that apply.

- **A.** Those units get zero gradient, so they are unlikely to recover on their own.
- **B.** A smaller learning rate, better initialization, or a leaky variant of ReLU can help.
- **C.** This is expected and harmless; ReLU is supposed to output zeros.
- **D.** The network's effective capacity has dropped.

<details>
<summary>Answer</summary>

**A, B, D**

ReLU's derivative is 0 for negative inputs, so a unit that is negative for every example never updates: a dead ReLU. Some zeros per example are normal (sparsity), but units that are zero for *all* examples are wasted capacity.

- **A:** Correct.
- **B:** Correct. Leaky ReLU keeps a small slope for negative inputs.
- **C:** Zeros for some inputs are fine; zeros for every input mean the unit is dead.
- **D:** Correct.

</details>

## 8. Reflection (hard)

A teammate's PyTorch classifier ends with `nn.Softmax(dim=1)` and is trained with `nn.CrossEntropyLoss`. Accuracy is mediocre and training is slow. What is wrong, and how do you fix it?

<details>
<summary>Answer</summary>

**Model answer.** `CrossEntropyLoss` expects raw logits and applies log-softmax internally. Applying softmax first means the loss sees softmax of probabilities: a squashed, nearly flat distribution with tiny gradients. Remove the final `nn.Softmax` from the model, train on logits, and apply softmax only at inference when you need probabilities.

Return logits from the model and let the loss function do the squashing; it is both correct and numerically stable.

</details>
