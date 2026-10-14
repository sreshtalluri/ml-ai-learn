---
title: GANs, generator versus discriminator
summary: Train a generator and a discriminator against each other, derive the optimal discriminator and the non-saturating loss, and diagnose mode collapse and vanishing gradients.
skill: deep-learning
minutes: 40
prerequisites: [autoencoders-diffusion-and-transfer, backpropagation, loss-functions]
related: [latent-diffusion-stable-diffusion, flow-matching-rectified-flows, logistic-regression, information-theory]
adapted_from: [phases/08-generative-ai/03-gans-generator-discriminator]
---

# GANs, generator versus discriminator

> **Mental model.** A forger and a detective train together. The forger (generator) turns random noise into fake samples. The detective (discriminator) looks at real and fake samples and says how likely each one is to be real. Every time the detective learns a new tell, the forger gets a gradient that says "move your fakes this way". If they reach a stalemate where the detective can only guess, the fakes are distributed like the real data.

**You will learn to**
- Write the GAN minimax objective and the two alternating training steps, with shapes.
- Derive the optimal discriminator $D^*(x) = p_{\text{data}}/(p_{\text{data}} + p_g)$ and explain the 0.5 equilibrium.
- Compute why the original generator loss saturates and the non-saturating loss doesn't.
- Implement a GAN with hand-written backprop and recognize mode collapse.
- Explain where adversarial training lives on in 2026: perceptual and adversarial losses in autoencoders and fast distilled diffusion samplers.

**Why it matters.** GANs introduced a way to learn a generator without writing down a likelihood: train a classifier to tell real from fake and use its gradient as the training signal. Diffusion and flow matching now produce the best open-domain images, but the adversarial idea stayed: Stable Diffusion's autoencoder is trained with an adversarial loss, and 1 to 4 step distilled samplers like SDXL-Turbo use one. GAN failure modes (mode collapse, a too-strong discriminator) are also classic interview material because they test whether you understand gradients, not just architectures.

## 1. Intuition

**No likelihood, just a critic.** An autoencoder trained with squared error averages over plausible outputs, which makes images blurry: the mean of many sharp faces is a blurry face. A GAN never measures pixel distance. Its generator $G$ only learns from whether the discriminator $D$ was fooled. A sample that looks plausible is rewarded, even if it matches no particular training image.

**Two networks, alternating steps.**

1. Train $D$ for one step as an ordinary binary classifier: real samples labeled 1, generated samples labeled 0.
2. Train $G$ for one step to make $D$ say "real" on fresh fakes. The gradient flows *through* $D$ (frozen for this step) into $G$.

**What the gradient means.** $D$'s output is a landscape over input space: high where it thinks real data lives. $G$'s update moves each fake sample uphill on that landscape. As $G$ improves, $D$ must redraw the landscape. At equilibrium $D$ outputs 0.5 everywhere the data lives, so there is no uphill left.

**Why it breaks.** The game has no guarantee of converging. Two common failures:

- **Vanishing gradient.** If $D$ becomes too confident early, the original generator loss gives almost no gradient. The fix is the *non-saturating* loss (section 3).
- **Mode collapse.** $G$ finds one region $D$ currently accepts and puts all its samples there. $D$'s gradient only acts where fakes are, so nothing pulls $G$ toward modes it never visits.

## 2. Visualization

![Left: histograms of generator samples during training on synthetic two-mode data at -2 and +2. At step 0 the samples are a broad blob around 0; at step 300 they spread from -4.5 to 4.5; at step 1000 two peaks form near -2 and +2; at step 3000 they match the real data. Right: final generators. The normal run covers both modes; a hard-start run whose generator began near x = 8 puts all its samples on the +2 mode, which is mode collapse.](../../figures/gans-generator-discriminator.png)

*Synthetic data: a 50/50 mixture of Gaussians at $-2$ and $+2$ with standard deviation 0.4. Both networks are 2-hidden-layer MLPs with 32 units, trained with Adam in NumPy. In the hard-start run, $G$'s output bias begins at 8 and $D$ is pre-trained for 300 steps: the fakes walk to the nearest real mode and stay there.*

**Try it** (predict first, then run the script and check):
1. In the normal run, $D(\text{real})$ and $D(\text{fake})$ stay near 0.50 the whole time. Predict whether that means $D$ is useless, and what $G$ would do if $D$ stopped training.
2. In the hard start, predict which mode the generator lands on and why it doesn't spread to the other one.
3. Change `g_start=8.0` to `g_start=-8.0`. Predict which mode wins now.
4. Call `train(1, non_saturating=False)` from the normal start. Predict whether it still learns, then do the same with the hard start.

## 3. The math

### Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $x$ | a real sample | $[d]$ (here $d = 1$) |
| $z$ | noise input to the generator, $z \sim \mathcal{N}(0, I)$ | $[d_z]$ (here 4) |
| $G_\theta(z)$ | generator; outputs a fake sample $\hat{x}$ | $[d]$ |
| $D_\psi(x)$ | discriminator; probability that $x$ is real, $\sigma(a)$ of a logit $a$ | scalar in $(0, 1)$ |
| $p_{\text{data}}, p_g$ | density of real data and of generator samples | functions of $x$ |
| $V(D, G)$ | the game value | scalar |

### The minimax game

```math
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z}\big[\log\big(1 - D(G(z))\big)\big]
```

The $D$ step maximizes $V$, which is the same as minimizing binary cross-entropy with real = 1 and fake = 0:

```math
L_D = -\log D(x) - \log\big(1 - D(G(z))\big)
```

### The optimal discriminator

For a fixed $G$, maximize $p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x))$ at each point $x$. Setting the derivative to 0, $p_{\text{data}}/D - p_g/(1 - D) = 0$, gives

```math
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
```

Substituting $D^*$ back in gives $V = -\log 4 + 2\,\text{JSD}(p_{\text{data}} \,\|\, p_g)$, where JSD is the Jensen-Shannon divergence. It is minimized, at $-\log 4$, exactly when $p_g = p_{\text{data}}$, and then $D^*(x) = 0.5$ everywhere.

### Saturating versus non-saturating generator loss

The original generator loss is $\log(1 - D(G(z)))$, to be minimized. The fix that Goodfellow et al. proposed in the same paper is to maximize $\log D(G(z))$ instead:

```math
L_G^{\text{sat}} = \log\big(1 - D(G(z))\big), \qquad L_G^{\text{NS}} = -\log D(G(z))
```

With $D = \sigma(a)$, the gradients with respect to the logit $a$ are

```math
\frac{\partial L_G^{\text{sat}}}{\partial a} = -D, \qquad \frac{\partial L_G^{\text{NS}}}{\partial a} = -(1 - D)
```

### Worked example: a confident discriminator

Early in training $D$ easily rejects a fake: $D(G(z)) = 0.01$, logit $a = \ln(0.01/0.99) = -4.595$.

- Saturating: $\partial L / \partial a = -0.01$.
- Non-saturating: $\partial L / \partial a = -(1 - 0.01) = -0.99$.

Both point the same way, but the non-saturating gradient is $0.99 / 0.01 = 99$ times larger, exactly when $G$ most needs to move.

### Worked example: one discriminator loss

$D(x) = 0.8$ on a real sample and $D(G(z)) = 0.3$ on a fake.

```math
L_D = -\ln 0.8 - \ln(1 - 0.3) = 0.2231 + 0.3567 = 0.5798
```

The non-saturating generator loss on that fake is $-\ln 0.3 = 1.204$. At equilibrium, with $D = 0.5$ everywhere, $L_D = -2\ln 0.5 = 1.386 = \ln 4$.

### Worked example: optimal D at one point

At some $x$, $p_{\text{data}}(x) = 0.3$ and $p_g(x) = 0.1$: $D^*(x) = 0.3 / (0.3 + 0.1) = 0.75$. $G$ is under-producing samples here, and the generator's gradient will push fakes toward this $x$.

## 4. Implementation

The full script is [`code/13-deep-architectures/gan_scratch.py`](../../code/13-deep-architectures/gan_scratch.py) (NumPy only, about 4 seconds on a laptop CPU).

### Build it from scratch

**Step 1: the data and a small MLP.** Real data is a two-mode mixture, shape `[n, 1]`. Both networks are the same MLP class: LeakyReLU hidden layers (no dead units for the generator's gradient to get stuck in) and a linear output. $G$ maps `[n, 4] -> [n, 1]`; $D$ maps `[n, 1] -> [n, 1]` logits.

```python
def sample_real(rng, n):
    return (np.where(rng.random(n) < 0.5, -2.0, 2.0) + 0.4 * rng.standard_normal(n))[:, None]

    def forward(self, x):
        p = self.p
        a1 = x @ p["W1"] + p["b1"]
        h1 = np.where(a1 > 0, a1, 0.2 * a1)
        a2 = h1 @ p["W2"] + p["b2"]
        h2 = np.where(a2 > 0, a2, 0.2 * a2)
        self.cache = (x, a1, h1, a2, h2)
        return h2 @ p["W3"] + p["b3"]
```

*Predict:* what range of values can $G$'s output take, and why doesn't $D$'s forward end in a sigmoid?

**Step 2: backprop that also returns the input gradient.** The generator's update needs $\partial L / \partial \hat{x}$, the gradient with respect to $D$'s *input*. So `backward` returns parameter gradients and `d1 @ W1.T`, the gradient flowing out of the first layer.

```python
    def backward(self, d_out):
        x, a1, h1, a2, h2 = self.cache
        p = self.p
        g = {"W3": h2.T @ d_out, "b3": d_out.sum(0)}
        d2 = (d_out @ p["W3"].T) * np.where(a2 > 0, 1, 0.2)
        g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ p["W2"].T) * np.where(a1 > 0, 1, 0.2)
        g["W1"], g["b1"] = x.T @ d1, d1.sum(0)
        return g, d1 @ p["W1"].T                  # parameter grads, input grad
```

*Predict:* the input gradient has which shape for a batch of 64 one-dimensional fakes?

**Step 3: the discriminator step.** Ordinary logistic regression on top of the MLP: stack real and fake, label 1 and 0. The gradient of binary cross-entropy with respect to the logit is `sigmoid(logit) - label`.

```python
def d_step(D, real, fake, lr):
    x = np.vstack([real, fake])
    y = np.vstack([np.ones_like(real), np.zeros_like(fake)])
    p = sigmoid(D.forward(x))
    g, _ = D.backward((p - y) / len(x))
    D.adam(g, lr)
    return p[: len(real)].mean(), p[len(real):].mean()
```

*Predict:* in this step, does any gradient reach $G$'s weights?

**Step 4: the generator step.** Fresh noise, a forward pass through $G$ then $D$, the logit gradient from section 3, back through $D$ to its input, then back through $G$. Only $G$'s optimizer steps; $D$'s weights are untouched.

```python
def g_step(G, D, z, lr, non_saturating=True):
    fake = G.forward(z)
    p = sigmoid(D.forward(fake))
    # non-saturating: minimise -log D(G(z))   -> dL/dlogit = p - 1
    # saturating:     minimise log(1 - D(G(z))) -> dL/dlogit = -p
    d_logit = (p - 1) if non_saturating else -p
    _, d_fake = D.backward(d_logit / len(z))      # gradient flows through D to its input...
    g, _ = G.backward(d_fake)                    # ...then into G (D is NOT updated)
    G.adam(g, lr)
```

*Predict:* if you accidentally also called `D.adam` on these gradients, which way would $D$ move?

**Step 5: the loop and a mode check.** One $D$ step, then one $G$ step, 3,000 times. Adam uses $\beta_1 = 0.5$ (a common GAN default since DCGAN), since a lot of momentum makes the chase oscillate. The mode check counts samples within 1 of each mode.

```python
        real = sample_real(rng, 64)
        fake = G.forward(rng.standard_normal((64, 4)))
        d_real, d_fake = d_step(D, real, fake, lr=1e-3)
        g_step(G, D, rng.standard_normal((64, 4)), lr=1e-3, non_saturating=non_saturating)

def coverage(xs):
    left = np.mean(np.abs(xs + 2) < 1)
    right = np.mean(np.abs(xs - 2) < 1)
    return left, right
```

*Predict:* a generator that outputs exactly $+2$ every time. What does `coverage` return, and what would a single-number "sample quality" metric say about it?

### Use the library

In PyTorch, autograd handles both backward passes. The two details to get right are `fake.detach()` in the $D$ step (so $D$'s loss doesn't update $G$) and using the "real" label in the $G$ step (the non-saturating loss).

```python
import torch, torch.nn as nn, torch.nn.functional as F

mlp = lambda i, o: nn.Sequential(nn.Linear(i, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, o))
G, D = mlp(4, 1), mlp(1, 1)
opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = F.binary_cross_entropy_with_logits                    # sigmoid + BCE, numerically stable

for real in loader:                                         # real: [B, 1]
    fake = G(torch.randn(len(real), 4))
    d_loss = bce(D(real), torch.ones_like(real)) + bce(D(fake.detach()), torch.zeros_like(real))
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    g_loss = bce(D(fake), torch.ones_like(real))            # non-saturating: -log D(G(z))
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()
```

For images, $G$ and $D$ become convolutional networks (DCGAN: strided convolutions, batch norm in $G$, LeakyReLU in $D$), and stabilizers like spectral normalization in $D$ or a gradient penalty are standard.

### Run it

```bash
uv run code/13-deep-architectures/gan_scratch.py
```

Key lines of output:

```text
saturating     loss log(1 - D): d/dlogit = -D       = -0.010
non-saturating loss -log D:     d/dlogit = -(1 - D) = -0.990  (99x larger)

== Non-saturating GAN, 3000 steps ==
 step | D(real) | D(fake)
    1 |    0.52 |    0.51
 3000 |    0.50 |    0.50
mode coverage: 0.45 near -2, 0.51 near +2 (data: 0.50 / 0.50 minus tails)

== Hard start: G's samples begin near x = 8, D pre-trained 300 steps, 1500 steps ==
seed | D(fake) at start | near -2 | near +2
   0 |            0.014 |    0.00 |    1.00
```

$D$ hovers near 0.5 because $G$ keeps up with it on this easy 1-D problem; the useful signal is in the small, shifting slopes of $D$, not its average. All three hard-start seeds collapse onto the $+2$ mode.

## 5. Engineering

**Where GANs stand in 2026.** For open-domain text-to-image, video, and audio, diffusion and flow-matching models are the default: they train stably and cover the whole distribution. GANs remain strong in narrow domains (faces, a single product category, super-resolution) where their one-pass sampling is fast. The adversarial *loss* is everywhere:

- **Autoencoders for latent diffusion.** Stable Diffusion's autoencoder is trained with reconstruction, perceptual (LPIPS), and patch-based adversarial losses so decoded images are sharp ([latent diffusion](05-latent-diffusion-stable-diffusion.md)).
- **Fast distilled samplers.** SDXL-Turbo uses Adversarial Diffusion Distillation (an adversarial loss plus score distillation from a teacher) to sample in 1 to 4 steps; SD3-Turbo uses a latent variant. Not every fast sampler is adversarial: Latent Consistency Models use consistency distillation.

**Stabilizers that matter.** Non-saturating loss always. Adam with $\beta_1$ of 0 to 0.5. Spectral normalization on $D$ (bounds its slope). Wasserstein loss with gradient penalty (WGAN-GP) when $D$ overpowers $G$. Two time-scale learning rates (TTUR: $D$'s learning rate higher than $G$'s). An exponential moving average of $G$'s weights for sampling. Data augmentation applied to both real and fake inputs of $D$ when data is small.

**Evaluation.** Losses don't tell you sample quality: they're a game, not a target. Use FID (Fréchet distance between Inception features of real and generated sets; lower is better) on enough samples (it is biased at small sample counts, so compare at a fixed count such as 50,000), precision and recall for generative models (quality versus coverage separately), and look at sample grids. Mode collapse shows up as low recall and repeated samples.

**Compute.** Sampling is one forward pass of $G$, so latency is low and fixed. Training is two networks, each step, and usually needs careful tuning per dataset.

> [!WARNING]
> A falling generator loss does not mean better samples. If $D$'s accuracy goes to 100% and $G$'s loss rises steadily, $G$ is getting no useful gradient; if $D$'s outputs sit at 0.5 but samples look identical, you have mode collapse. Always look at samples and a coverage metric.

### Common mistakes

- Forgetting `detach()` on fakes in the $D$ step, so $D$'s loss also trains $G$ in the wrong direction.
- Reusing the $D$-step fakes for the $G$ step after $D$ has updated, which gives stale gradients. Generate fresh fakes or recompute.
- Using the saturating loss $\log(1 - D(G(z)))$.
- Letting batch norm see a batch of only real or only fake samples in $D$ and leak batch statistics; prefer spectral norm or layer norm in $D$.
- Judging progress from loss curves instead of samples, FID, and coverage.

## 6. Knowledge check

<!-- quiz:gans-generator-discriminator -->
**[Take the GAN quiz](../../quizzes/gans-generator-discriminator.md)**
<!-- /quiz -->

**Practice exercise.** At a point $x$, $p_{\text{data}}(x) = 0.05$ and $p_g(x) = 0.20$. What is $D^*(x)$? If a fake sits at this $x$ and $D$ outputs $D^*$, what are the saturating and non-saturating gradients with respect to the logit?

<details>
<summary>Solution</summary>

$D^*(x) = 0.05 / (0.05 + 0.20) = 0.2$. $G$ is over-producing here. Saturating: $\partial L/\partial a = -D = -0.2$. Non-saturating: $-(1 - D) = -0.8$, four times larger. Both push the logit up (making the sample look more real), which in practice moves the fake toward regions where $D$ is higher, away from this over-crowded spot.

</details>

**Implementation challenge.** Extend the script to 2-D data: 8 Gaussians on a ring of radius 2. Count how many of the 8 modes have at least 5% of samples after 1,000, 3,000, and 6,000 steps. Then add minibatch standard deviation (append the batch's average standard deviation as an extra feature to $D$'s input) and compare.

<details>
<summary>Hint</summary>

`sample_real` becomes angle $2\pi k/8$ for random $k$, plus small Gaussian noise; $D$ takes `[n, 2]`, $G$ outputs `[n, 2]`. For the minibatch feature, compute `s = x.std(0).mean()` and concatenate a column filled with `s`. Its gradient flows to every sample in the batch, so the backward pass needs a small extra term. If that's fiddly, stop the gradient through `s` first and see whether it still helps $D$ spot collapsed batches.

</details>

## Summary

- A GAN trains $G$ to fool $D$ and $D$ to separate real from fake; $G$ learns from the gradient of $D$, never from a likelihood.
- For fixed $G$, $D^* = p_{\text{data}}/(p_{\text{data}} + p_g)$; at the global optimum $p_g = p_{\text{data}}$ and $D^* = 0.5$.
- Use the non-saturating loss $-\log D(G(z))$; its logit gradient $-(1-D)$ is large exactly when $D$ is confident.
- Mode collapse happens because $D$ only shapes gradients where fakes already are; watch coverage and samples, not loss.
- In 2026 adversarial losses live on inside autoencoders and fast distilled diffusion samplers; diffusion and flow matching are the default generators.

**Next:** [Latent diffusion and Stable Diffusion](05-latent-diffusion-stable-diffusion.md)

**Related:** [Autoencoders, diffusion, and transfer learning](03-autoencoders-diffusion-and-transfer.md) · [Flow matching and rectified flows](06-flow-matching-rectified-flows.md) · [Loss functions](../10-neural-networks/02-loss-functions.md) · [Backpropagation](../11-gradient-descent-backprop/02-backpropagation.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Conditional GANs & Pix2Pix](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/04-conditional-gans-pix2pix)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [GANs — Generator vs Discriminator](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/03-gans-generator-discriminator).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Walk me through how a GAN trains. What is each network optimizing?</strong></summary>

There are two networks. The generator maps noise z to a sample. The discriminator maps a sample to the probability that it's real. Training alternates. In the D step, D is an ordinary binary classifier: real samples labeled 1, generated samples labeled 0, minimizing binary cross-entropy, with the fakes detached so G isn't updated. In the G step, fresh fakes go through G and then D, and the loss is minus log D of G of z, which is minimized when D says real. The gradient flows back through D, whose weights are frozen for that step, into G. Formally it's a minimax game: for fixed G, the best D is p_data over p_data plus p_g, and plugging that in gives minus log 4 plus twice the Jensen-Shannon divergence, minimized when the generator's distribution equals the data's, where D outputs 0.5.

</details>

<details>
<summary><strong>Why does the original generator loss saturate, and what does the non-saturating loss change?</strong></summary>

The original game has G minimize log of one minus D of G of z. With D as a sigmoid of a logit a, the gradient with respect to a is minus D. Early in training, D easily rejects fakes, so D of G of z is near zero, and the gradient is near zero too: G learns almost nothing exactly when it's worst. The non-saturating loss instead minimizes minus log D of G of z. Its gradient with respect to the logit is minus one plus D, which is close to minus one when D is confident. At D equal to 0.01, the gradients are minus 0.01 versus minus 0.99, a 99 times difference, in the same direction. Both losses have the same fixed point, so you lose nothing in theory and gain a usable signal in practice. Goodfellow's original paper already recommended it.

</details>

<details>
<summary><strong>Your GAN's samples all look nearly identical, but the losses look stable. Diagnose and fix.</strong></summary>

That's mode collapse. Stable losses don't contradict it: G found a region D currently accepts and D's average output sits near 0.5. It happens because D only shapes gradients where fakes are; nothing pulls G toward modes it never visits, and G can hop between modes as D catches up. Confirm it with a coverage measure: precision and recall for generative models, FID at a fixed sample count, nearest-neighbor checks, or for toy data, counting samples per mode. Fixes, roughly in order: make D see batch-level statistics (minibatch standard deviation), stabilize D with spectral normalization or a gradient penalty, switch to a Wasserstein-style loss, lower G's learning rate or use two time-scale updates, average G's weights, and add augmentation if data is small. If coverage matters more than one-step speed, consider diffusion or flow matching instead.

</details>

<details>
<summary><strong>GAN, diffusion, or flow matching for a product that must generate 512 by 512 product photos in under 50 ms on one GPU?</strong></summary>

The constraint is latency: 50 milliseconds rules out a 30-step diffusion sampler at that resolution on typical hardware, since each step is a full network pass. Options. A GAN in a narrow domain (one product category, fixed backgrounds) gives a single forward pass and can be very sharp, but training is fiddly and coverage of rare variants is weaker. Or train a diffusion or flow-matching model, which is more stable and covers the distribution, then distill it to 1 to 4 steps with adversarial diffusion distillation or consistency distillation, so the adversarial loss is used as a distillation tool. I'd prototype the distilled latent model first, measure latency per step and FID plus human ratings against the GAN, and pick on measured quality and coverage at the 50 ms budget, not on reputation.

</details>
