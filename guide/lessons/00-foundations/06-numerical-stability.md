---
title: Numerical stability
summary: Spot overflow, underflow, and cancellation before they turn a training run into NaN, and fix them with log-sum-exp, log-softmax, the right float format, loss scaling, and gradient checks.
skill: math
minutes: 40
prerequisites: [information-theory, calculus-for-ml, python-toolkit]
related: [loss-functions, intro-to-pytorch, debugging-neural-networks, training-and-regularization]
adapted_from: [phases/01-math-foundations/13-numerical-stability]
---

# Numerical stability

> **Mental model.** A float is a number written in scientific notation with a fixed number of digits and a fixed range of exponents. Most of the time that is invisible. It bites in three places: a number gets too big (overflow to `inf`), too small (underflow to 0), or you subtract two nearly equal numbers and only rounding noise is left. Every stable ML formula is a rewrite that keeps values away from those three edges.

**You will learn to**
- Read a float format's range and precision (fp32, fp16, bf16) and predict where it overflows or underflows.
- Derive the log-sum-exp trick and use it for stable softmax and cross-entropy from logits.
- Recognize catastrophic cancellation and rearrange a formula to avoid it.
- Explain why bf16 usually trains without loss scaling and fp16 does not.
- Check an analytic gradient with centered finite differences and read the relative error.

**Why it matters.** Loss turning into `NaN` hours into a run is one of the most common deep-learning failures, and a frequent interview question. Nearly every case traces back to one `exp`, `log`, division, or low-precision cast. The fixes are a few lines each, and they are built into the library functions you should be calling.

## 1. Intuition

Imagine a calculator that shows only 7 significant digits and can't display anything above $10^{38}$. That is float32.

**Overflow.** Ask it for $e^{100}$ (about $2.7 \times 10^{43}$) and it shows `inf`. Softmax computes $e^{z}$ for every logit, so a logit of 100 breaks it. But softmax only cares about *differences* between logits: adding the same number to every logit doesn't change the output. Subtract the largest logit first and the biggest exponent becomes $e^0 = 1$. Same answer, no overflow.

**Underflow.** Ask for $e^{-200}$ and it shows 0. Now take the log of that 0 to compute a loss and you get $-\infty$. The fix is to never leave log space: compute the log-probability directly as "logit minus log-sum-exp", which is a modest number like −200, instead of a probability that rounds to 0.

**Cancellation.** With 7 digits, 10001.0 and 10004.0 are stored fine, but their squares (around $10^8$) are stored only to the nearest 8. Subtract two such squares to get a variance and the rounding error is bigger than the answer. Center the data first and the problem disappears.

**Precision formats.** Half-precision formats trade digits for speed and memory. fp16 keeps more digits but tops out at 65,504. bf16 keeps fewer digits but has float32's exponent range. Training spikes often exceed 65,504, and small gradients fall below fp16's smallest value, so bf16 is the usual training choice on hardware that supports it.

## 2. Visualization

![Left: log-sum-exp of three float32 logits against the largest logit; the naive version is a dashed line that stops at about 88.3 where exp overflows, while the stable version continues straight to 120. Middle: log-log plot of gradient-check error against step size h; error falls as h shrinks until about 1e-5 for the centered difference and 1e-7 for the forward difference, then rises again as rounding noise takes over. Right: bars comparing float32, float16, bfloat16; log10 of max value is 38.5, 4.8, 38.5 and decimal digits are 6.9, 3.0, 2.1.](../../figures/numerical-stability.png)

*Synthetic inputs. Left: the naive formula is exactly right until it suddenly isn't. Middle: finite differences have a sweet spot; too large an $h$ is inaccurate, too small is drowned by rounding. Right: bf16 matches float32's range but has the fewest digits.*

**Try it** (predict, then run the script and check):
1. In the left panel, the naive line dies near 88.3, not 88.7 (where `exp` alone overflows in float32). Why does summing three exponentials move the cliff left?
2. In the middle panel, the centered error bottoms out near $h = 10^{-5}$. Predict the bottom for float32 inputs instead of float64: larger or smaller $h$?
3. Cast 70,000 to fp16 and to bf16 in the script. Predict both results before running.
4. Change the loss scale from 1024 to 65,536 and add a gradient of 1.0. Predict what happens to it in fp16.

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $z$ | logits | vector of length $K$ |
| $m$ | $\max_j z_j$ | scalar |
| $\text{LSE}(z)$ | log-sum-exp, $\log\sum_j e^{z_j}$ | scalar |
| $y$ | true class index | integer |
| $\varepsilon_{\text{mach}}$ | machine epsilon: gap between 1 and the next float | scalar |
| $h$ | finite-difference step | scalar |
| $S$ | loss-scale factor | scalar |

### Float formats

A float stores $(-1)^{s} \times 1.f \times 2^{e - \text{bias}}$. Exponent bits set the **range**, fraction (mantissa) bits set the **precision**. Values below from `np.finfo` (bf16 from its bit layout):

| Format | Exponent bits | Fraction bits | Max | Smallest normal | $\varepsilon_{\text{mach}}$ | Decimal digits |
|---|---|---|---|---|---|---|
| float64 | 11 | 52 | $1.8 \times 10^{308}$ | $2.2 \times 10^{-308}$ | $2.2 \times 10^{-16}$ | ~16 |
| float32 | 8 | 23 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | $1.2 \times 10^{-7}$ | ~7 |
| float16 | 5 | 10 | 65,504 | $6.1 \times 10^{-5}$ | $9.8 \times 10^{-4}$ | ~3 |
| bfloat16 | 8 | 7 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | $7.8 \times 10^{-3}$ | ~2 |

Below the smallest normal, **subnormal** numbers fill the gap with lost precision; fp16's smallest positive subnormal is $2^{-24} \approx 6 \times 10^{-8}$. Anything smaller rounds to 0.

Thresholds for `exp` in float32: $e^{88.72} \approx 3.4 \times 10^{38}$ is the largest that fits, and $e^{-103.3} \approx 1.4 \times 10^{-45}$ is the smallest subnormal; below that `exp` returns 0.

### The log-sum-exp trick

For any constant $c$:

```math
\begin{aligned}
\log\sum_j e^{z_j} &= \log\sum_j e^{z_j - c}\,e^{c} = \log\Big(e^{c}\sum_j e^{z_j - c}\Big) = c + \log\sum_j e^{z_j - c}
\end{aligned}
```

Choose $c = m = \max_j z_j$. Then every exponent $z_j - m \le 0$, so no term exceeds 1 (no overflow), and the term for the max equals $e^0 = 1$, so the sum is at least 1 and its log is at least 0 (no $\log 0$).

The same shift gives stable softmax, $\text{softmax}(z)_i = e^{z_i - m} / \sum_j e^{z_j - m}$, and stable cross-entropy straight from logits:

```math
\log \text{softmax}(z)_y = z_y - \text{LSE}(z), \qquad \text{loss} = \text{LSE}(z) - z_y
```

### Worked example 1: softmax of [100, 101, 102] in float32

Naive: $e^{100} \approx 2.69 \times 10^{43}$, which is larger than float32's max of $3.4 \times 10^{38}$, so all three exponentials are `inf` and `inf / inf` is `NaN`.

Stable: $m = 102$, shifted logits $[-2, -1, 0]$.

```math
\begin{aligned}
e^{-2}, e^{-1}, e^{0} &= 0.1353,\ 0.3679,\ 1.0000 \qquad \text{sum} = 1.5032 \\
\text{softmax} &= 0.1353/1.5032,\ 0.3679/1.5032,\ 1/1.5032 = 0.0900,\ 0.2447,\ 0.6652 \\
\text{LSE} &= 102 + \ln 1.5032 = 102 + 0.4076 = 102.4076
\end{aligned}
```

### Worked example 2: cross-entropy when the softmax underflows

Logits $[0, 200]$, true class 0. The stable softmax gives class 0 the probability $e^{-200}/(1 + e^{-200})$. $e^{-200} \approx 1.4 \times 10^{-87}$, below float32's smallest subnormal, so it rounds to 0 and $-\log 0 = \infty$.

From logits: $\text{LSE} = 200 + \ln(1 + e^{-200}) = 200 + 0 = 200$, so the loss is $200 - 0 = 200$. Correct and finite. The model is badly wrong, and the loss says so with a usable gradient.

### Worked example 3: catastrophic cancellation

Variance of $[10001, 10002, 10003, 10004]$ in float32. The true variance is 1.25.

One-pass formula $\mathbb{E}[x^2] - \mathbb{E}[x]^2$: both terms are about $1.0005 \times 10^{8}$. float32 near $10^8$ can only represent multiples of 8 (since $2^{26} \approx 6.7 \times 10^7$ and the 23-bit fraction gives a spacing of $2^{26-23} = 8$). Their difference is rounding noise; the script prints **−8.0**, a negative variance.

Two-pass: subtract the mean 10002.5 first, giving $[-1.5, -0.5, 0.5, 1.5]$. Squares $2.25, 0.25, 0.25, 2.25$, mean $5/4 = 1.25$. Exact.

### Worked example 4: loss scaling in fp16

Gradients $[2 \times 10^{-8},\ 10^{-7},\ 2 \times 10^{-4}]$. fp16's smallest subnormal is $5.96 \times 10^{-8}$:

- $2 \times 10^{-8}$ is less than half of that, so it rounds to **0**: the update is lost.
- $10^{-7}$ rounds to $2 \times 5.96 \times 10^{-8} = 1.19 \times 10^{-7}$: a 19% error.

Multiply the loss by $S = 1024$ before backward, so every gradient is 1024 times larger: $2.05 \times 10^{-5}$ and $1.02 \times 10^{-4}$, both well inside fp16's range. Cast, then divide by 1024 in float32: $2.0023 \times 10^{-8}$ and $1.0000 \times 10^{-7}$. Both recovered. If any scaled gradient becomes `inf`, skip the step and halve $S$ (dynamic loss scaling).

### Gradient checking

The centered difference has error $O(h^2)$; the forward difference has $O(h)$:

```math
\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}, \qquad
\text{rel. error} = \frac{\max|g_{\text{analytic}} - g_{\text{numeric}}|}{\max(|g_{\text{analytic}}|, |g_{\text{numeric}}|)}
```

Too small an $h$ makes $f(x+h)$ and $f(x-h)$ nearly equal, and cancellation takes over: total error is roughly $h^2 + \varepsilon_{\text{mach}}/h$, minimized near $h \approx \varepsilon_{\text{mach}}^{1/3} \approx 6 \times 10^{-6}$ in float64. Run checks in float64. Rule of thumb: relative error below $10^{-7}$ is right, around $10^{-3}$ or more is a bug. In the script, the gradient of LSE (which is softmax) checks to $2.5 \times 10^{-11}$; a deliberate 1% bug in one component shows up as $2.2 \times 10^{-3}$.

## 4. Implementation

### Build it from scratch

Start with the naive and stable versions side by side. The only change is subtracting the max, which the math above proved doesn't change the result.

```python
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax_stable(z):
    e = np.exp(z - z.max())        # largest exponent is exp(0) = 1: cannot overflow
    return e / e.sum()
```

*Predict:* what does `softmax_naive` return for `[-1000, -1001]`? (Hint: overflow isn't the only failure.)

Log-sum-exp gets the same shift, plus it adds the max back outside the log.

```python
def logsumexp_naive(z):
    return np.log(np.exp(z).sum())

def logsumexp_stable(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())   # sum >= 1, so the log is never log(0)
```

*Predict:* for `z = [-1000, -1001]`, the naive version returns `-inf`. What does the stable one return? (Answer: $-1000 + \ln(1 + e^{-1}) = -999.6867$.)

Cross-entropy is where it matters most. Even a stable softmax can round a tiny probability to 0 before the log; computing log-softmax directly never forms the probability.

```python
def cross_entropy_naive(z, y):
    return -np.log(softmax_stable(z)[y])     # even a stable softmax can underflow to 0 here

def cross_entropy_stable(z, y):
    return -(z[y] - logsumexp_stable(z))     # log-softmax: never takes log of a probability
```

*Predict:* for logits `[0, 200]` and true class 1, do the two versions agree? (Yes, both give about 0: the danger is only for the class with the tiny probability.)

NumPy has no bfloat16, so emulate it: bf16 is the top 16 bits of a float32. Add half of the dropped part (with round-to-nearest-even) and mask.

```python
def to_bf16(x):
    """Round float32 to bfloat16 (keep the top 16 bits, round to nearest even), return as float32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)
```

*Predict:* `to_bf16(1 + 1/256)` returns exactly 1.0. Why? (bf16 has 7 fraction bits, so the gap above 1 is $2^{-7} = 1/128$, and $1/256$ is exactly halfway: ties go to the even neighbour, 1.0.)

Last, the gradient checker: perturb one coordinate at a time in both directions.

```python
def numerical_grad(fn, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (fn(xp) - fn(xm)) / (2 * h)   # centered difference: error O(h^2)
    return g

def rel_error(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12)
```

*Predict:* this costs two function evaluations per parameter. How long would it take for a 7-billion-parameter model? (Which is why you gradient-check small layers, not whole models.)

### Use the library

```python
import torch, torch.nn.functional as F
torch.logsumexp(z, dim=-1)                 # the trick above
F.log_softmax(z, dim=-1)                   # z - logsumexp(z), never forms probabilities
F.cross_entropy(logits, target)            # log_softmax + NLL fused; pass LOGITS, not probabilities
F.binary_cross_entropy_with_logits(z, y)   # stable sigmoid + BCE (use instead of sigmoid then BCE)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # bf16 mixed precision, no scaler
    loss = model(x).loss
scaler = torch.amp.GradScaler("cuda")      # fp16 needs dynamic loss scaling
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.autograd.gradcheck(fn, inputs_double)                      # finite-difference check in float64
torch.autograd.set_detect_anomaly(True)    # debugging only: reports the op that produced NaN
```

Mixed precision keeps a float32 master copy of the weights and runs matmuls in the low-precision type; reductions like softmax and loss are kept in float32 by autocast.

### Run it

```bash
uv run code/00-foundations/numerical_stability.py
```

Key output:

```
float16  max=6.55e+04  min normal=6.104e-05  eps=0.0009766
bfloat16 max=3.39e+38  min normal=1.175e-38  eps=0.007812
variance of [10001.0, 10002.0, 10003.0, 10004.0] in float32: E[x^2]-E[x]^2 = -8.0000, two-pass = 1.2500
exp(z) = [inf inf inf]  -> naive softmax = [nan nan nan]
z - max = [-2. -1.  0.]  -> stable softmax = [0.09   0.2447 0.6652]
-log(softmax): inf   log-softmax: 200.0000
               70000.0: fp16                      inf  bf16 70144.0
cast to fp16      [0.0000000e+00 1.1920929e-07 2.0003319e-04]
x1024,fp16,/1024   [2.0023435e-08 1.0000076e-07 2.0003319e-04]
analytic [0.176542 0.032251 0.791207]  numeric [0.176542 0.032251 0.791207]  rel error 2.49e-11
with a 1% bug: rel error 2.23e-03
```

## 5. Engineering

**Use fused, logits-in functions.** `F.cross_entropy(logits, y)` and `binary_cross_entropy_with_logits` are stable. `log(softmax(x))` and `sigmoid` followed by `binary_cross_entropy` are not.

**Choosing a precision.** On GPUs and TPUs with bf16 support (NVIDIA Ampere and later, TPUs, recent Apple and AMD hardware), bf16 mixed precision is the default for training: float32 range, no loss scaler. fp16 needs dynamic loss scaling. Inference increasingly uses fp16, bf16, or lower (int8, fp8, 4-bit) depending on hardware; see [quantization](../../glossary.md#quantization). Keep optimizer states and the master weights in float32.

**Epsilons.** Normalization layers divide by $\sqrt{\sigma^2 + \epsilon}$ with $\epsilon$ around $10^{-5}$; Adam divides by $\sqrt{v} + \epsilon$ with $\epsilon = 10^{-8}$. An $\epsilon$ of $10^{-8}$ is below fp16's smallest normal and rounds away, which is one reason these ops stay in float32.

**Gradient clipping.** Clip the global gradient norm (commonly 1.0 for transformers) so one bad batch can't throw the weights into overflow territory. Log the pre-clip norm: a rising trend is an early warning.

**Determinism.** Floating-point addition is not associative, so GPU reductions in different orders give slightly different results. Expect ~$10^{-6}$ relative differences across hardware; compare with tolerances, not `==`.

> [!WARNING]
> **Failure modes.** Loss becomes `NaN` (overflow in `exp`, `log(0)`, division by ~0, or LR too high); loss stuck at $\ln K$ from vanishing gradients or a broken label pipeline; small silent accuracy loss from fp16 gradient underflow without a scaler; a negative variance from one-pass formulas; flaky tests from `==` on floats.

### Common mistakes

- Passing probabilities to a loss that expects logits (or the reverse). Double softmax trains, but badly.
- Clamping probabilities with `log(p + 1e-8)` and calling it stable. It hides the bug and caps the gradient; compute in log space instead.
- Running gradient checks in float32, then chasing "errors" of $10^{-4}$ that are just rounding.
- Casting the whole model to fp16 (not mixed precision) and losing small updates entirely.

## 6. Knowledge check

<!-- quiz:numerical-stability -->
**[Take the numerical stability quiz](../../quizzes/numerical-stability.md)**
<!-- /quiz -->

**Practice exercise.** Compute $\text{LSE}([3, 1, -2])$ by hand with the max trick, then the cross-entropy for true class 1.

<details>
<summary>Solution</summary>

$m = 3$. Shifted: $[0, -2, -5]$. Exponentials: $1, 0.1353, 0.0067$. Sum: $1.1420$. $\ln 1.1420 = 0.1328$, so $\text{LSE} = 3.1328$.

Cross-entropy for class 1: $\text{LSE} - z_1 = 3.1328 - 1 = 2.1328$ nats. Check: softmax of class 1 is $0.1353/1.1420 = 0.1185$, and $-\ln 0.1185 = 2.133$.

</details>

**Implementation challenge.** Write a stable `log_sigmoid(x)` that is correct for $x = \pm 1000$ in float64, and a stable `bce_with_logits(x, y)` built on it. Test against the naive formula at $x = \pm 30$.

<details>
<summary>Solution</summary>

$\log\sigma(x) = -\log(1 + e^{-x})$. For large negative $x$, $e^{-x}$ overflows. Use $\log\sigma(x) = \min(x, 0) - \log(1 + e^{-|x|})$, which only exponentiates a non-positive number.

```python
def log_sigmoid(x):
    return np.minimum(x, 0) - np.log1p(np.exp(-np.abs(x)))

def bce_with_logits(x, y):
    # -[y log σ(x) + (1-y) log σ(-x)], using log σ(-x) = log σ(x) - x
    return -(y * log_sigmoid(x) + (1 - y) * (log_sigmoid(x) - x))
```

`log_sigmoid(-1000)` is $-1000$, `log_sigmoid(1000)` is $0$. `np.log1p` keeps precision when $e^{-|x|}$ is tiny. At $x = 30$, `1 - sigmoid(30)` is about $9.4 \times 10^{-14}$ in float64 and loses most of its digits; in float32 it is exactly 0 and the naive BCE for $y = 0$ is infinite, while `bce_with_logits(30, 0)` returns 30.0.

</details>

## Summary

- Floats have finite range and precision. `exp` overflows past about 88.7 in float32; probabilities underflow to 0; subtracting nearby numbers leaves noise.
- Log-sum-exp: subtract the max, add it back outside the log. Softmax and cross-entropy from logits use it.
- Compute losses in log space from logits; never `log(softmax(x))`.
- bf16 keeps float32's range with fewer digits and usually trains without loss scaling; fp16 needs a dynamic loss scaler.
- Check gradients with centered differences in float64; relative error below about $10^{-7}$ is a pass.

**Next:** [The language of ML](../01-ml-vocabulary/01-ml-vocabulary.md)

**Related:** [Information theory](05-information-theory.md) · [Loss functions](../10-neural-networks/02-loss-functions.md) · [Intro to PyTorch](../10-neural-networks/03-intro-to-pytorch.md) · [Debugging neural networks](../12-training-regularization/02-debugging-neural-networks.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Numerical Stability](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/13-numerical-stability).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Your loss becomes NaN after a few thousand steps. How do you find the cause?</strong></summary>

First find where the NaN starts. Log the loss, the global gradient norm before clipping, and the max absolute activation per layer every step. If the gradient norm or the logits grow steadily before the NaN, it's divergence: lower the learning rate, add warmup, add or tighten gradient clipping. If the jump is sudden, find the batch: save the step's inputs, replay it, and turn on anomaly detection (set_detect_anomaly) to get the op that produced the first NaN or inf. The usual culprits are a hand-written log of a softmax or sigmoid output that hit 0, a division by a norm or variance that hit 0, an exp of a large value, or fp16 overflow above 65,504 in an activation. Fixes: use fused logits-in losses, add the epsilon inside the sqrt, use bf16 or a dynamic loss scaler. Also check the data for NaN or extreme values in that batch.

</details>

<details>
<summary><strong>Why do we subtract the max logit in softmax? Doesn't that change the answer?</strong></summary>

It doesn't change the answer, because softmax is shift-invariant: multiplying numerator and denominator by e^−c cancels, so softmax(z − c) equals softmax(z) for any constant c. Choosing c as the max makes every exponent zero or negative, so no term exceeds 1 and nothing overflows, and the largest term is exactly 1, so the denominator is at least 1 and never underflows to zero. The same identity gives log-sum-exp: log Σ e^z = m + log Σ e^(z−m). It matters in practice because float32's exp overflows just past 88.7. Logits of [100, 101, 102] give inf over inf, which is NaN, while the shifted version gives 0.090, 0.245, 0.665. For the loss, go one step further and use log-softmax, z_y minus LSE, so you never take the log of a probability that rounded to zero.

</details>

<details>
<summary><strong>fp16 or bf16 for training, and why?</strong></summary>

bf16 where the hardware supports it. Both are 16 bits, but they split them differently. fp16 has 5 exponent bits and 10 fraction bits: about 3 decimal digits, but a max of 65,504 and a smallest subnormal near 6e−8. bf16 has float32's 8 exponent bits and 7 fraction bits: about 2 digits, but the same range as float32. Training cares more about range. Activation spikes exceed 65,504 and overflow in fp16, and many gradients are smaller than 6e−8 and flush to zero, which silently stops learning. fp16 therefore needs dynamic loss scaling: multiply the loss by a large S, unscale gradients, and halve S whenever an inf appears. bf16 usually needs none. Either way, keep master weights, optimizer states, and reductions like softmax and the loss in float32. fp16 is still fine for inference, where values are bounded and the extra digits help.

</details>

<details>
<summary><strong>You implemented a custom layer's backward pass. How do you check it, and what step size do you use?</strong></summary>

Compare against centered finite differences: for each input coordinate, (f(x+h) − f(x−h)) / 2h, where f is the scalar loss. Do it in float64 on a small random input. The centered form has error proportional to h², but too small an h makes f(x+h) and f(x−h) nearly equal, and the subtraction cancels to rounding noise. Total error behaves like h² + ε/h, which is smallest near the cube root of machine epsilon, about 6e−6 in float64, so h between 1e−5 and 1e−6 is typical. Report the relative error, max |analytic − numeric| over the larger magnitude. Below 1e−7 is a pass; around 1e−3 or larger is a bug. In my test, a deliberate 1% error in one component showed up as 2.2e−3. Avoid kinks like ReLU at 0 in the test points, since the derivative is undefined there. PyTorch's gradcheck automates this.

</details>
