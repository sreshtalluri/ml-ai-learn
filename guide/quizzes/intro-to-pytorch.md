<!-- GENERATED from intro-to-pytorch.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Intro to PyTorch

Covers the lesson [Intro to PyTorch](../lessons/10-neural-networks/03-intro-to-pytorch.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/intro-to-pytorch/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What is autograd in PyTorch?

- **A.** A model-compression tool
- **B.** An automatic differentiation engine that records operations on tensors and computes gradients by running the chain rule backwards
- **C.** A hyperparameter tuning framework
- **D.** A data-loading library

<details>
<summary>Answer</summary>

**B.** An automatic differentiation engine that records operations on tensors and computes gradients by running the chain rule backwards

Tensors with `requires_grad=True` record the operations applied to them; `backward()` walks that record in reverse.

- **A:** Compression (pruning, quantization) is separate from gradient computation.
- **B:** Correct.
- **C:** Autograd computes gradients; it doesn't search hyperparameters.
- **D:** Data loading is `torch.utils.data`.

</details>

## 2. Multiple choice (easy)

What does `torch.no_grad()` do?

- **A.** It prevents overfitting
- **B.** It enables GPU acceleration
- **C.** It stops autograd from recording operations, saving memory and time when you don't need gradients
- **D.** It freezes the model's weights permanently

<details>
<summary>Answer</summary>

**C.** It stops autograd from recording operations, saving memory and time when you don't need gradients

No graph is built inside the context, so intermediate activations aren't kept for a backward pass.

- **A:** It has no regularizing effect.
- **B:** Device placement is `.to(device)`.
- **C:** Correct.
- **D:** Weights are frozen per parameter with `requires_grad_(False)`; no_grad only affects operations inside the block.

</details>

## 3. Multiple choice (medium)

What does `nn.Linear(784, 256)` create?

- **A.** A batch-normalization layer
- **B.** A ReLU activation
- **C.** A dropout layer with rate 784/256
- **D.** A fully connected layer with a weight of shape (256, 784) and a bias of shape (256,)

<details>
<summary>Answer</summary>

**D.** A fully connected layer with a weight of shape (256, 784) and a bias of shape (256,)

Weights are stored as [out_features, in_features] and the layer computes $xW^\top + b$.

- **A:** Batch norm is `nn.BatchNorm1d`.
- **B:** ReLU is `nn.ReLU` and has no parameters.
- **C:** Dropout is `nn.Dropout(p)` with p between 0 and 1.
- **D:** Correct.

</details>

## 4. Calculation (medium)

How many parameters does `nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))` have?

<details>
<summary>Answer</summary>

**203530**

$784 \times 256 + 256 = 200{,}960$. $256 \times 10 + 10 = 2{,}570$. ReLU has none. Total $203{,}530$.

</details>

## 5. Calculation (medium)

With $x = 2$, $w = 3$, $b = 1$, $y = 10$ and $L = (wx + b - y)^2$, you call `backward()` on two separately computed copies of $L$ without zeroing. What is `w.grad`?

<details>
<summary>Answer</summary>

**-24**

Each backward gives $\partial L/\partial w = 2(wx + b - y)\,x = 2 \times (-3) \times 2 = -12$, and gradients add into `.grad`, so $-12 + (-12) = -24$.

</details>

## 6. Arrange in order (easy)

Order the steps of one training iteration.

- `optimizer.step()`
- `loss.backward()`
- `optimizer.zero_grad()`
- Loss: `loss = F.cross_entropy(logits, y)`
- Forward: `logits = model(x)`

<details>
<summary>Answer</summary>

1. Forward: `logits = model(x)`
2. Loss: `loss = F.cross_entropy(logits, y)`
3. `optimizer.zero_grad()`
4. `loss.backward()`
5. `optimizer.step()`

zero_grad can also come first; what matters is that it runs before backward and that step comes after backward.

</details>

## 7. Multiple choice (hard)

Your model has dropout. Validation accuracy changes every time you evaluate the same checkpoint on the same data. What's the most likely bug?

- **A.** The validation set is too small
- **B.** `model.eval()` isn't called before evaluating, so dropout is still randomly zeroing units
- **C.** `torch.no_grad()` is missing
- **D.** The optimizer state wasn't saved

<details>
<summary>Answer</summary>

**B.** `model.eval()` isn't called before evaluating, so dropout is still randomly zeroing units

In train mode dropout is active, so the same input gives different outputs. `model.eval()` turns it off (and makes batch norm use running statistics).

- **A:** A small set gives a noisy estimate, but the same set and weights should give the same number every time.
- **B:** Correct.
- **C:** Missing no_grad wastes memory but doesn't change the outputs.
- **D:** Optimizer state matters for resuming training, not for evaluating fixed weights.

</details>

## 8. Multiple choice (hard)

Training with plain SGD, the loss drops for a few steps and then oscillates wildly and diverges. The gradient norm grows every step. Which missing line best explains this?

- **A.** `model.train()`
- **B.** `optimizer.zero_grad()`
- **C.** `torch.manual_seed(0)`
- **D.** `loss.item()`

<details>
<summary>Answer</summary>

**B.** `optimizer.zero_grad()`

Without zero_grad, each step's `.grad` is the sum of all previous gradients, so updates keep growing.

- **A:** Modules start in train mode by default, and the mode doesn't make gradients grow.
- **B:** Correct.
- **C:** Seeding affects reproducibility, not stability.
- **D:** `.item()` is for logging; omitting it doesn't change the gradients.

</details>

## 9. Multiple choice (medium)

Why is a PyTorch training loop so much faster than the scratch `Value` engine?

- **A.** It uses smaller data types by default
- **B.** It uses a different differentiation algorithm
- **C.** It skips the backward pass
- **D.** Each tensor operation runs as one optimized C++/CUDA kernel over a whole batch, while `Value` runs interpreted Python per number

<details>
<summary>Answer</summary>

**D.** Each tensor operation runs as one optimized C++/CUDA kernel over a whole batch, while `Value` runs interpreted Python per number

Both use reverse-mode differentiation over a recorded graph. The difference is batching and compiled kernels.

- **A:** The default is float32 in both cases (Python floats are 64-bit, but that isn't the main factor).
- **B:** Both are reverse-mode autodiff.
- **C:** Training always needs the backward pass.
- **D:** Correct.

</details>

## 10. Multiple choice (hard)

Your GPU fits a batch of 32 but the recipe calls for an effective batch of 128. What's the standard approach?

- **A.** Use a learning rate 4 times smaller with batch 32
- **B.** Gradient accumulation - call `backward()` on `loss / 4` for 4 micro-batches, then `step()` and `zero_grad()` once
- **C.** Wrap training in `torch.no_grad()` to save memory
- **D.** Call `model.eval()` to disable dropout and save memory

<details>
<summary>Answer</summary>

**B.** Gradient accumulation - call `backward()` on `loss / 4` for 4 micro-batches, then `step()` and `zero_grad()` once

Accumulating gradients over 4 micro-batches reproduces the batch-128 gradient (exactly, for layers without batch statistics), at the cost of 4 forward/backward passes per step.

- **A:** That changes the optimization dynamics; it doesn't reproduce a batch-128 gradient.
- **B:** Correct.
- **C:** With no_grad there is no graph, so `backward()` fails.
- **D:** eval mode changes layer behaviour; it doesn't meaningfully cut memory and makes training wrong.

</details>
