<!-- GENERATED from debugging-neural-networks.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Debugging neural networks

Covers the lesson [Debugging neural networks](../lessons/12-training-regularization/02-debugging-neural-networks.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/debugging-neural-networks/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

Why is debugging a neural network harder than debugging ordinary code?

- **A.** There are no debugging tools for neural networks
- **B.** A broken network usually runs without errors and produces plausible-looking numbers, so bugs are silent
- **C.** Neural networks are written in special languages
- **D.** Neural networks can't be tested

<details>
<summary>Answer</summary>

**B.** A broken network usually runs without errors and produces plausible-looking numbers, so bugs are silent

Shape broadcasting, shuffled labels, or a dead layer all run fine and still print a loss.

- **A:** Gradient checks, hooks, anomaly detection, and experiment trackers all exist.
- **B:** Correct.
- **C:** They are ordinary Python (or C++/CUDA underneath).
- **D:** Every check in this lesson is a test; they just test behaviour, not exact outputs.

</details>

## 2. Calculation (easy)

What step-0 cross-entropy loss do you expect from a well-initialized 10-class classifier? Give 3 decimals.

<details>
<summary>Answer</summary>

**2.303** (within ±0.002)

A near-uniform prediction gives $-\ln(1/10) = \ln 10 = 2.303$.

</details>

## 3. Multiple choice (easy)

What is the overfit-one-batch test?

- **A.** Train on the full dataset until it overfits
- **B.** Tune hyperparameters on the validation set until it overfits
- **C.** Use a very large batch size
- **D.** Train repeatedly on one small batch with regularization off and check the loss goes near zero

<details>
<summary>Answer</summary>

**D.** Train repeatedly on one small batch with regularization off and check the loss goes near zero

Any correct model with enough capacity can memorize a handful of examples. If it can't, the loss, backward pass, optimizer wiring, or input-label pairing is broken.

- **A:** That costs a full run and confounds data issues with plumbing issues.
- **B:** Overfitting the validation set is a mistake, not a test.
- **C:** Batch size isn't the point; the batch is deliberately small.
- **D:** Correct.

</details>

## 4. Multiple choice (medium)

A run's loss becomes NaN after a few steps. What is the most likely cause?

- **A.** The model has too few parameters
- **B.** The activation function is wrong
- **C.** The learning rate is too high (or a log/division hit zero), so values overflow
- **D.** The dataset is too small

<details>
<summary>Answer</summary>

**C.** The learning rate is too high (or a log/division hit zero), so values overflow

Overshooting updates blow up weights and logits; overflow or log(0) then produces inf and NaN. Lower the LR 10×, use fused logits-in losses, add clipping.

- **A:** A small model underfits; it doesn't produce NaN.
- **B:** Standard activations don't produce NaN on their own.
- **C:** Correct.
- **D:** Small data causes overfitting, not NaN.

</details>

## 5. Multiple choice (hard)

Training loss falls steadily, but validation loss is flat from the very first epoch. What does this point to?

- **A.** A train/validation mismatch, for example different preprocessing, a leak in training features, or a distribution shift
- **B.** The learning rate is too low
- **C.** The model needs more layers
- **D.** Classic overfitting

<details>
<summary>Answer</summary>

**A.** A train/validation mismatch, for example different preprocessing, a leak in training features, or a distribution shift

Overfitting shows validation improving first and then getting worse. Flat from the start means what's learned on train doesn't transfer at all, which usually means the two sets aren't processed or drawn the same way.

- **A:** Correct.
- **B:** A low LR would slow training loss too.
- **C:** Capacity isn't the issue when training loss is falling.
- **D:** Overfitting has an initial period where validation improves.

</details>

## 6. Multiple choice (medium)

The loss doesn't move at all from step 0. What do you check first?

- **A.** Try a different architecture
- **B.** Switch optimizers
- **C.** That gradients are nonzero and reach the parameters, the optimizer holds the model's parameters, the LR isn't tiny, and the loss depends on the model's output
- **D.** Collect more data

<details>
<summary>Answer</summary>

**C.** That gradients are nonzero and reach the parameters, the optimizer holds the model's parameters, the LR isn't tiny, and the loss depends on the model's output

A perfectly flat loss usually means updates aren't happening, which is a wiring problem. Architecture and data come later.

- **A:** Changing the architecture before checking the plumbing adds unknowns.
- **B:** A different optimizer won't help if gradients never reach the weights.
- **C:** Correct.
- **D:** More data doesn't help a model that isn't updating.

</details>

## 7. Calculation (medium)

A hidden layer has weight norm 8.0 and gradient norm 0.16, trained with SGD at learning rate 0.05. What is the update-to-weight ratio?

<details>
<summary>Answer</summary>

**0.001** (within ±0.00005)

$r = \eta\lVert\nabla W\rVert / \lVert W\rVert = 0.05 \times 0.16 / 8.0 = 0.008 / 8.0 = 0.001$, right at the common heuristic.

</details>

## 8. Multiple choice (hard)

Two runs both stall near $\ln K$. Run A's layer 2 has 97% of units off for every example and gradient norms around $10^{-4}$. Run B's per-layer stats look healthy and it can overfit one batch. Which diagnosis fits?

- **A.** Both have dead ReLUs
- **B.** A has dead ReLUs (a model/initialization problem); B most likely has a data problem such as shuffled or misaligned labels
- **C.** A has shuffled labels; B has dead ReLUs
- **D.** Both have learning rates that are too low

<details>
<summary>Answer</summary>

**B.** A has dead ReLUs (a model/initialization problem); B most likely has a data problem such as shuffled or misaligned labels

Sick internals point to the model; healthy internals plus the ability to memorize point to the data having no learnable signal.

- **A:** B's units are alive and its gradients are normal.
- **B:** Correct.
- **C:** Shuffled labels don't kill units; dead units don't depend on labels.
- **D:** A low LR would show small updates in every layer, not dead units, and B can overfit a batch.

</details>

## 9. Multiple choice (medium)

An LR range test shows the loss lowest at learning rate 1.0, exploding just above it. What learning rate do you start with?

- **A.** 1.0, the minimum
- **B.** About 0.1, roughly 10× below the minimum, then confirm with a couple of short runs
- **C.** 10, since higher is faster
- **D.** 0.0001, to be safe

<details>
<summary>Answer</summary>

**B.** About 0.1, roughly 10× below the minimum, then confirm with a couple of short runs

The minimum sits right next to the divergence point and is biased by training progress during the sweep, so back off about 10×.

- **A:** Right at the edge, small changes in data or schedule push it into divergence.
- **B:** Correct.
- **C:** The sweep already showed that range diverges.
- **D:** Four orders of magnitude below wastes most of the training budget.

</details>

## 10. Multiple choice (hard)

Your hand-written backward pass gives a gradient-check relative error of 1.1 on hidden layers but $10^{-7}$ on the output layer. Where is the bug most likely?

- **A.** In the softmax cross-entropy gradient
- **B.** In how gradients pass back through the hidden activations, for example a missing ReLU mask
- **C.** In the finite-difference step size
- **D.** Nowhere; errors around 1 are normal with ReLU

<details>
<summary>Answer</summary>

**B.** In how gradients pass back through the hidden activations, for example a missing ReLU mask

The output layer's gradient comes straight from the loss, and it checks out. Errors appear only once the gradient passes back through an activation, so the activation's backward is wrong.

- **A:** If that were wrong, the output layer would fail too.
- **B:** Correct.
- **C:** A bad step size gives errors around $10^{-4}$ in every layer, not exactly where activations are.
- **D:** ReLU kinks add small errors on rare entries, not errors of order 1 across a layer.

</details>
