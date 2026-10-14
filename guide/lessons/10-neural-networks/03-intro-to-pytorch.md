---
title: Intro to PyTorch
summary: Build a tiny autograd engine, then use PyTorch tensors, autograd, nn.Module, and the five-line training loop to train a network on CPU, and recognize the bugs that loop invites.
skill: deep-learning
minutes: 50
prerequisites: [loss-functions, backpropagation, python-toolkit]
related: [gradient-descent, debugging-neural-networks, numerical-stability]
adapted_from: [phases/03-deep-learning-core/11-intro-to-pytorch]
---

# Intro to PyTorch

> **Mental model.** PyTorch is NumPy that keeps a receipt. Every operation on a tensor that needs gradients is written onto a receipt (the computational graph). When you call `loss.backward()`, PyTorch reads the receipt backwards and applies the chain rule at each line, filling in `.grad` for every parameter. Everything else (layers, optimizers, data loaders) is convenience built on that one idea.

**You will learn to**
- Build a scalar autograd engine and check its gradients against PyTorch.
- Create and reshape tensors, and predict shapes, dtypes, and broadcasting results.
- Explain `requires_grad`, `.backward()`, gradient accumulation, and `torch.no_grad()`.
- Define a model with `nn.Module` and count its parameters.
- Write the training loop (forward, loss, zero_grad, backward, step) and the evaluation loop, and say what `model.train()` and `model.eval()` change.

**Why it matters.** PyTorch is the default framework for deep-learning research and most production training, and the Hugging Face ecosystem is built on it. Coding interviews for ML roles often ask you to write a training loop from memory, and the most common bugs (forgetting `zero_grad`, forgetting `eval()`, applying softmax before the loss) all live in about ten lines. Knowing what autograd actually does makes those bugs obvious.

## 1. Intuition

In [backpropagation](../11-gradient-descent-backprop/02-backpropagation.md) you computed gradients by hand: run the forward pass, then walk backwards multiplying local derivatives. That is mechanical, so a computer can do it if it remembers what happened.

PyTorch remembers. A **tensor** is an n-dimensional array, like a NumPy array, that can live on a CPU or GPU. If a tensor has `requires_grad=True` (all model parameters do), every operation on it creates a new tensor that records *which operation* made it and *from which inputs*. Chained together, these records form a graph from the parameters to the loss.

`loss.backward()` walks that graph from the loss back to the parameters. At each node it applies that operation's local derivative, multiplies by the gradient flowing in (the chain rule), and adds the result into the inputs' `.grad`. Note the word **adds**: gradients accumulate. That's handy (you can sum gradients over several small batches) and dangerous (if you forget to reset them, each step uses the sum of all previous gradients).

On top of autograd:
- **`nn.Module`** is a container that registers its parameters, so `model.parameters()` finds them all.
- **An optimizer** (SGD, AdamW) holds references to those parameters and, on `step()`, updates each one using its `.grad`.
- **A `DataLoader`** cuts the dataset into shuffled mini-batches.

The training loop is the same five steps every time: forward, loss, clear old gradients, backward, step.

## 2. Visualization

![Left: training loss over 40 epochs for the same model and seed; with zero_grad every step the loss falls smoothly from 0.45 to 0.04; with zero_grad forgotten it jumps up to nearly 2 in the first epochs and then wanders between 0.3 and 0.7. Right: the trained network's decision boundary curving between two interleaved half-moon classes, with validation accuracy 1.00.](../../figures/intro-to-pytorch.png)

*Synthetic two-moons data, 480 training and 120 validation points. Left: one missing line (`opt.zero_grad()`) and every step uses the running sum of all earlier gradients. Right: what the correct loop learned, a curved boundary no linear model could draw.*

**Try it** (predict, then run the script):
1. Without `zero_grad`, the effective gradient is a sum that keeps growing. AdamW divides by a running estimate of gradient size, which partly hides this. Predict what happens if you switch the optimizer to plain `torch.optim.SGD(..., lr=0.1)`.
2. Remove `model.eval()` from `accuracy`. Predict whether validation accuracy changes from run to run, and why.
3. Change `hidden=32` to `hidden=4`. Predict the new parameter count before running.
4. Print `model(xtr).shape` and `ytr.shape`. Why does `F.cross_entropy` accept a `[480, 2]` float tensor and a `[480]` integer tensor?

## 3. The math

This lesson is about a mechanism, so this section walks one tiny computation through autograd with every number.

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $x$ | input | scalar here; $[B, d_{\text{in}}]$ in a model |
| $w, b$ | parameters (`requires_grad=True`) | scalars here |
| $\hat{y}$ | prediction $wx + b$ | scalar |
| $y$ | target | scalar |
| $L$ | loss $(\hat{y} - y)^2$ | scalar |
| $\bar{v}$ | the gradient $\partial L / \partial v$ stored in `v.grad` | same shape as $v$ |
| $\eta$ | learning rate | scalar |

### Forward pass: build the graph

With $x = 2$, $w = 3$, $b = 1$, $y = 10$, autograd records four nodes:

| Node | Operation | Value | Local derivatives |
|---|---|---|---|
| $u$ | $w \times x$ | $3 \times 2 = 6$ | $\partial u/\partial w = x = 2$ |
| $\hat{y}$ | $u + b$ | $6 + 1 = 7$ | $\partial \hat{y}/\partial u = 1$, $\partial \hat{y}/\partial b = 1$ |
| $e$ | $\hat{y} - y$ | $7 - 10 = -3$ | $\partial e/\partial \hat{y} = 1$ |
| $L$ | $e^2$ | $9$ | $\partial L/\partial e = 2e = -6$ |

### Backward pass: walk it in reverse

Start with $\bar{L} = 1$. Each step multiplies the incoming gradient by the local derivative:

```math
\begin{aligned}
\bar{e} &= \bar{L} \cdot 2e = 1 \times (-6) = -6 \\
\bar{\hat{y}} &= \bar{e} \cdot 1 = -6 \\
\bar{b} &= \bar{\hat{y}} \cdot 1 = -6, \qquad \bar{u} = \bar{\hat{y}} \cdot 1 = -6 \\
\bar{w} &= \bar{u} \cdot x = -6 \times 2 = -12
\end{aligned}
```

So `w.grad = -12` and `b.grad = -6`, which both the scratch engine and PyTorch print.

### Accumulation and the update

Call backward again on a freshly computed loss without clearing, and each `.grad` gets the new gradient **added**: $\bar{w} = -12 + (-12) = -24$. That's what `zero_grad()` prevents.

One SGD step with $\eta = 0.01$ and the correct gradients:

```math
w \leftarrow 3 - 0.01 \times (-12) = 3.12, \qquad b \leftarrow 1 - 0.01 \times (-6) = 1.06
```

New prediction $3.12 \times 2 + 1.06 = 7.30$, new loss $(7.30 - 10)^2 = 7.29$, down from 9.

### Shapes in the real model

The script's `MLP` maps $[B, 2] \to [B, 32] \to [B, 32] \to [B, 2]$. `nn.Linear(in, out)` stores a weight of shape $[\text{out}, \text{in}]$ and computes $x W^\top + b$. Parameters:

```math
(2 \times 32 + 32) + (32 \times 32 + 32) + (32 \times 2 + 2) = 96 + 1056 + 66 = 1218
```

`F.cross_entropy(logits, y)` takes logits $[B, K]$ (float) and class indices $[B]$ (int64), and returns the mean loss, a scalar, which is what `.backward()` needs. At step 0 it should be near $\ln 2 = 0.693$ for 2 classes; the script prints 0.682.

## 4. Implementation

### Build it from scratch

A `Value` holds a number, its gradient, the values it came from, and a function that pushes its gradient to those parents. Addition and multiplication each record their local derivatives.

```python
class Value:
    """A number that remembers how it was computed, so it can send gradients back."""

    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad = data, 0.0
        self._parents, self._backward = parents, backward

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward():                       # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out
```

*Predict:* why `+=` rather than `=` in `backward`? (Hint: what if the same value is used twice, as in `a + a`?)

Multiplication swaps in the other factor; power uses the power rule. Subtraction is built from the other two.

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward():                       # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __sub__(self, other):
        return self + (other * -1 if isinstance(other, Value) else -other)

    def __pow__(self, k):
        out = Value(self.data**k, (self,))
        def backward():                       # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = backward
        return out
```

*Predict:* for `c = a * b` with `a = 3`, `b = 2`, and `c.grad = 1`, what are `a.grad` and `b.grad`?

`backward()` sorts the graph so each node runs after every node that uses it, seeds the loss with gradient 1, and walks in reverse.

```python
    def backward(self):
        order, seen = [], set()
        def visit(v):                         # topological order: parents before children
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0                       # dL/dL = 1
        for v in reversed(order):             # walk from the loss back to the leaves
            v._backward()
```

*Predict:* what goes wrong if you call `L.backward()` twice on the same `L`? (Every intermediate `.grad` still holds its first-pass value, so the second pass adds on top at every node: `w.grad` ends at −60, not −24. PyTorch frees the graph after `backward()`, so a second call raises an error unless you pass `retain_graph=True`.)

That is all of autograd. PyTorch does the same thing on tensors, with each operation's backward written in C++ or CUDA.

### Use the library

Same graph, same numbers:

```python
wt = torch.tensor(3.0, requires_grad=True)
bt = torch.tensor(1.0, requires_grad=True)
Lt = (wt * 2.0 + bt - 10.0) ** 2
Lt.backward()                                # wt.grad = -12, bt.grad = -6
```

A model is an `nn.Module`. Layers assigned as attributes are registered, so `model.parameters()` finds their weights. The last layer outputs raw logits, because `F.cross_entropy` applies log-softmax itself (see [loss functions](02-loss-functions.md)).

```python
class MLP(nn.Module):
    def __init__(self, d_in=2, hidden=32, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_out),           # raw logits: no softmax here
        )

    def forward(self, x):                       # x: [batch, d_in] -> logits: [batch, d_out]
        return self.net(x)
```

The training loop, five numbered steps. `model.train()` turns dropout (and batch-norm statistics updates) on.

```python
def train(model, loader, epochs, lr=1e-2, zero_grad=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        model.train()                           # dropout on
        total = 0.0
        for xb, yb in loader:
            logits = model(xb)                  # 1. forward
            loss = F.cross_entropy(logits, yb)  # 2. loss
            if zero_grad:
                opt.zero_grad()                 # 3. clear old gradients
            loss.backward()                     # 4. backward: fill .grad
            opt.step()                          # 5. update parameters
            total += loss.item() * len(xb)
        losses.append(total / len(loader.dataset))
    return losses
```

Evaluation turns off both dropout (`model.eval()`) and graph recording (`torch.no_grad()`, here as a decorator), which saves memory and time.

```python
@torch.no_grad()
def accuracy(model, x, y):
    model.eval()                                # dropout off
    return (model(x).argmax(1) == y).float().mean().item()
```

`.item()` pulls a Python number out of a one-element tensor; use it for logging so you don't keep the graph alive by storing the loss tensor. Save weights with `torch.save(model.state_dict(), path)` and restore with `model.load_state_dict(torch.load(path, weights_only=True))`.

**Beyond CPU.** Move the model and every batch to the same device: `device = "cuda" if torch.cuda.is_available() else "cpu"`, then `model.to(device)` and `xb.to(device)` (Apple silicon uses `"mps"`). `torch.compile(model)` (PyTorch 2.x) traces and fuses the model into faster kernels without changing your code; for real training add mixed precision with `torch.autocast` (see [numerical stability](../00-foundations/06-numerical-stability.md)).

### Run it

```bash
uv run code/10-neural-networks/intro_to_pytorch.py
```

PyTorch is declared in the script's header (PEP 723 inline metadata), so `uv` installs it into a cached environment on first run; that download takes a minute or two. Key output:

```
L = 9.0, dL/dw = -12.0, dL/db = -6.0
L = 9.0, dL/dw = -12.0, dL/db = -6.0
after a second backward without zeroing: dL/dw = -24.0 (gradients accumulate)
X + bias =
tensor([[10., 21., 32.],
        [13., 24., 35.]])
weight (3, 2), bias (3,)  (out_features, in_features)
parameters: 1218  (2*32+32 + 32*32+32 + 32*2+2)
step-0 loss 0.6815 (ln 2 = 0.6931)
correct loop:      final train loss 0.0406, val accuracy 1.000
no zero_grad():    final train loss 0.4439, val accuracy 0.775
train mode, same input twice -> identical? False;  eval mode -> identical? True
```

Exact losses can differ slightly across PyTorch versions and CPUs; the pattern won't.

## 5. Engineering

**Where the time goes.** PyTorch is fast because each tensor operation runs as one optimized kernel over the whole batch. Python loops over individual numbers (like the scratch `Value`) are millions of times slower. Keep work in batched tensor ops, and keep the GPU busy: `DataLoader(num_workers=..., pin_memory=True)` overlaps data loading with compute.

**Memory.** Autograd keeps every intermediate activation until backward, so activation memory grows with batch size and depth. Use `torch.no_grad()` (or `torch.inference_mode()`) for evaluation, and gradient checkpointing for very deep models. Accumulating `total += loss` (a tensor) instead of `loss.item()` keeps every step's graph alive and leaks memory.

**Reproducibility.** Seed `torch.manual_seed`, the DataLoader's `generator`, and NumPy/Python if used. On GPU, some kernels are non-deterministic; `torch.use_deterministic_algorithms(True)` trades speed for repeatability.

**Gradient accumulation on purpose.** To simulate a batch of 256 on a GPU that fits 64, call `backward()` on `loss / 4` for four micro-batches, then `step()` and `zero_grad()` once. This is the useful side of accumulation.

**Checkpoints.** Save `model.state_dict()`, `optimizer.state_dict()`, the scheduler, the step, and the RNG states, so a resumed run continues exactly. Load with `weights_only=True` for files you didn't create; arbitrary pickles can run code.

> [!WARNING]
> **Failure modes.** Forgetting `opt.zero_grad()` (gradients accumulate across steps); forgetting `model.eval()` (dropout stays on and batch norm uses batch statistics at inference); softmax before `F.cross_entropy`; model on GPU and data on CPU (a device mismatch error); calling `.numpy()` on a tensor that requires grad (use `.detach()` first); in-place operations on tensors autograd still needs.

### Common mistakes

- Putting `zero_grad()` in the wrong place (it must run before each `backward()`, not between `backward()` and `step()`).
- Thinking `torch.no_grad()` and `model.eval()` do the same thing. One stops graph recording; the other changes layer behaviour. Inference needs both.
- Using integer tensors as model inputs (`torch.arange(6)` is int64; `nn.Linear` needs float).
- Leaving `shuffle=True` on the validation loader and then comparing per-batch metrics across runs.

## 6. Knowledge check

<!-- quiz:intro-to-pytorch -->
**[Take the PyTorch quiz](../../quizzes/intro-to-pytorch.md)**
<!-- /quiz -->

**Practice exercise.** With $x = 1$, $w = 2$, $b = -1$, $y = 4$, compute $L = (wx + b - y)^2$ and both gradients by hand. Then compute $w$ and $b$ after one SGD step with $\eta = 0.1$.

<details>
<summary>Solution</summary>

$\hat{y} = 2 \times 1 - 1 = 1$. $e = 1 - 4 = -3$. $L = 9$.

$\partial L/\partial e = 2e = -6$. $\bar{b} = -6$. $\bar{w} = -6 \times x = -6$.

Update: $w = 2 - 0.1 \times (-6) = 2.6$; $b = -1 - 0.1 \times (-6) = -0.4$. New prediction $2.6 - 0.4 = 2.2$, new loss $(2.2 - 4)^2 = 3.24$.

</details>

**Implementation challenge.** Add a `relu()` method to `Value`, build a 2-input neuron $\text{relu}(w_1 x_1 + w_2 x_2 + b)$, and check all three gradients against PyTorch for one input where the ReLU is active and one where it is not.

<details>
<summary>Solution</summary>

```python
def relu(self):
    out = Value(max(0.0, self.data), (self,))
    def backward():
        self.grad += (self.data > 0) * out.grad      # slope 1 if active, else 0
    out._backward = backward
    return out
Value.relu = relu
```

For an active input, each weight's gradient is its input and the bias gradient is 1 (times the upstream gradient). For an inactive input, all three are 0, in both engines. That zero is why a neuron whose input is negative for every example (a "dead ReLU") stops learning.

</details>

## Summary

- A tensor with `requires_grad=True` records the operations applied to it; `backward()` runs the chain rule over that record and adds into `.grad`.
- Gradients accumulate, so call `zero_grad()` once per step before `backward()`.
- `nn.Module` registers parameters; `nn.Linear(in, out)` has weight shape `[out, in]`.
- The loop is forward, loss, zero_grad, backward, step. Evaluation uses `model.eval()` and `torch.no_grad()`.
- Output raw logits and let `F.cross_entropy` apply the softmax.

**Next:** [Gradient descent](../11-gradient-descent-backprop/01-gradient-descent.md)

**Related:** [Backpropagation](../11-gradient-descent-backprop/02-backpropagation.md) · [Loss functions](02-loss-functions.md) · [Debugging neural networks](../12-training-regularization/02-debugging-neural-networks.md) · [Numerical stability](../00-foundations/06-numerical-stability.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Introduction to PyTorch](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/11-intro-to-pytorch).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Write a PyTorch training loop from memory and explain each line.</strong></summary>

Set model.train(), then loop over batches from the DataLoader. Move the batch to the model's device. Forward: logits = model(x). Loss: F.cross_entropy(logits, y), which takes raw logits and integer labels. Then optimizer.zero_grad(), because backward adds into .grad and we want only this batch's gradient. Then loss.backward(), which walks the recorded graph and fills .grad for every parameter. Optionally clip with clip_grad_norm_. Then optimizer.step(), which updates each parameter from its gradient and its optimizer state. Log loss.item(), not the tensor, so the graph is freed. After each epoch, evaluate under model.eval() and torch.no_grad(): eval switches dropout off and makes batch norm use its running statistics, and no_grad stops recording the graph to save memory. Save the model and optimizer state_dicts at checkpoints. If there's a scheduler, step it at the granularity it was designed for.

</details>

<details>
<summary><strong>What's the difference between model.eval() and torch.no_grad()?</strong></summary>

They control different things, and inference usually needs both. model.eval() sets a flag on every submodule that changes layer behaviour: dropout stops dropping units, and batch normalization uses the running mean and variance gathered in training instead of the current batch's statistics. It does nothing to gradients. torch.no_grad() is a context that stops autograd from recording operations, so no graph is built, intermediate activations aren't kept, and memory and time go down; it does nothing to layer behaviour. Forget eval() and predictions are noisy (dropout) or depend on who else is in the batch (batch norm), so validation numbers drift between runs. Forget no_grad() and evaluation still gives correct numbers but uses much more memory and can run out on large models. torch.inference_mode() is a stricter, slightly faster version of no_grad for pure inference.

</details>

<details>
<summary><strong>Training loss is noisy and creeps upward over time, while the same code with a different optimizer seemed fine. What do you check first?</strong></summary>

I'd check whether zero_grad() is called every step, and in the right place, before backward. Without it, each step's .grad is the sum of every gradient so far, so updates grow and point in a stale averaged direction, giving noisy, rising loss. Adaptive optimizers like Adam partly hide this because they divide by a running estimate of the gradient's size, which explains why changing the optimizer changed the symptom. Quick test: print the global gradient norm each step; with the bug it grows steadily. Next I'd check the learning rate and warmup, that the loss uses logits and not probabilities, and that model.train() and model.eval() are toggled correctly, since dropout left on in eval would make validation noisy too. The same accumulation is useful when done on purpose for gradient accumulation, but then zero_grad runs once every k micro-batches and the loss is divided by k.

</details>

<details>
<summary><strong>How many parameters does nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)) have, and how much memory does training it with AdamW take in float32?</strong></summary>

nn.Linear(in, out) has an out × in weight plus out biases. First layer: 784 × 256 = 200,704 weights plus 256 biases, 200,960. Second: 256 × 10 = 2,560 plus 10, 2,570. ReLU has none. Total 203,530 parameters. In float32 that's 4 bytes each, about 0.81 MB for the weights. Training also stores one gradient per parameter, another 0.81 MB, and AdamW keeps two moment estimates per parameter, 1.63 MB. So weights, gradients, and optimizer state come to 16 bytes per parameter, about 3.3 MB. Activations are extra and scale with batch size: for batch 64, the hidden layer holds 64 × 256 values per stored tensor, which is small here. For a 7-billion-parameter model the same 16-bytes-per-parameter rule gives 112 GB before activations, which is why large models use mixed precision, sharded optimizer states, or parameter-efficient fine-tuning.

</details>
