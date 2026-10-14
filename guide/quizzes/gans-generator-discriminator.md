<!-- GENERATED from gans-generator-discriminator.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: GANs, generator versus discriminator

Covers the lesson [GANs, generator versus discriminator](../lessons/13-deep-architectures/04-gans-generator-discriminator.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/gans-generator-discriminator/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

What does the discriminator step of GAN training optimize?

- **A.** Mean squared error between fakes and the nearest real sample
- **B.** Binary cross-entropy with real samples labeled 1 and generated samples labeled 0
- **C.** The likelihood of the real data under the generator
- **D.** The KL divergence between the noise and the data

<details>
<summary>Answer</summary>

**B.** Binary cross-entropy with real samples labeled 1 and generated samples labeled 0

D is an ordinary binary classifier. Its loss is $-\log D(x) - \log(1 - D(G(z)))$.

- **A:** GANs never measure pixel distance to a target; that's what makes them sharp.
- **C:** GANs avoid computing a likelihood; that's the point of the adversarial setup.
- **D:** No term compares the noise distribution with the data.

</details>

## 2. Calculation (medium)

At a point $x$, $p_{\text{data}}(x) = 0.3$ and $p_g(x) = 0.1$. What is the optimal discriminator output $D^*(x)$?

<details>
<summary>Answer</summary>

**0.75** (within ±0.001)

$D^*(x) = p_{\text{data}}/(p_{\text{data}} + p_g) = 0.3/0.4 = 0.75$.

</details>

## 3. Calculation (medium)

$D(G(z)) = 0.02$. With $D = \sigma(a)$, how many times larger in magnitude is the non-saturating generator gradient $\partial(-\log D)/\partial a$ than the saturating one $\partial \log(1 - D)/\partial a$?

<details>
<summary>Answer</summary>

**49** (within ±0.01)

Saturating gives $-D = -0.02$; non-saturating gives $-(1 - D) = -0.98$. Ratio $0.98/0.02 = 49$.

</details>

## 4. Calculation (medium)

At the GAN equilibrium, $D$ outputs 0.5 on every real and fake sample. What is the discriminator loss $L_D = -\ln D(x) - \ln(1 - D(G(z)))$? Give 3 decimals.

<details>
<summary>Answer</summary>

**1.386** (within ±0.002)

$-\ln 0.5 - \ln 0.5 = 0.693 + 0.693 = 1.386 = \ln 4$.

</details>

## 5. Multiple choice (medium)

Why is the non-saturating loss $-\log D(G(z))$ preferred over minimizing $\log(1 - D(G(z)))$?

- **A.** It has a different optimum that gives sharper samples
- **B.** Early on, D rejects fakes confidently, and the original loss then has almost zero gradient; the non-saturating loss has a large gradient there
- **C.** It makes D train faster
- **D.** It removes the need for a discriminator

<details>
<summary>Answer</summary>

**B.** Early on, D rejects fakes confidently, and the original loss then has almost zero gradient; the non-saturating loss has a large gradient there

Both share the same fixed point. The difference is gradient magnitude when $D(G(z))$ is near 0.

- **A:** Both losses share the same equilibrium.
- **C:** It changes only G's loss; D's step is unchanged.
- **D:** The loss is computed through D.

</details>

## 6. Multiple choice (hard)

In a PyTorch GAN, you forget `fake.detach()` in the discriminator step and step only D's optimizer. Then the G step zeroes G's gradients before its own backward. What is the main effect?

- **A.** D's loss gradients are also computed for G's parameters, which wastes compute; correctness is saved only because G's gradients are zeroed before its step
- **B.** D stops learning
- **C.** G receives no gradient at all in its own step
- **D.** Training becomes deterministic

<details>
<summary>Answer</summary>

**A.** D's loss gradients are also computed for G's parameters, which wastes compute; correctness is saved only because G's gradients are zeroed before its step

Without detach, backprop from D's loss runs through G too. If G's grads weren't zeroed, G would be pushed to make fakes look fake. Detach makes intent explicit and saves compute.

- **B:** D's own gradients are unaffected by the missing detach.
- **C:** G's step recomputes its loss and backward from scratch.
- **D:** Detach has nothing to do with randomness.

</details>

## 7. Multiple choice (hard)

A GAN trained on 10 digit classes produces mostly 1s and 7s. D's average output is near 0.5 and losses look stable. What's happening and what helps?

- **A.** Underfitting; train longer with the same settings
- **B.** Mode collapse; D only shapes gradients where fakes are. Add minibatch statistics to D, spectral norm or a gradient penalty, and measure coverage (recall, FID)
- **C.** The dataset has too many 1s and 7s
- **D.** D is too weak; remove spectral normalization

<details>
<summary>Answer</summary>

**B.** Mode collapse; D only shapes gradients where fakes are. Add minibatch statistics to D, spectral norm or a gradient penalty, and measure coverage (recall, FID)

Stable losses don't rule out collapse. Measure coverage and give D batch-level information so it can penalize low diversity.

- **A:** More steps often lets G hop between modes rather than cover them.
- **C:** Possible, but check the data's class balance before blaming it; the symptom matches collapse.
- **D:** Removing a stabilizer usually makes collapse worse.

</details>

## 8. Select all that apply (medium)

Which are standard ways to stabilize GAN training? Select all that apply.

- **A.** Spectral normalization on the discriminator
- **B.** Wasserstein loss with gradient penalty
- **C.** Adam with a low beta1 such as 0.5 or 0
- **D.** Using the saturating generator loss

<details>
<summary>Answer</summary>

**A, B, C**

Spectral norm and gradient penalties bound D's slope; low momentum reduces oscillation. The saturating loss makes early training worse.

</details>

## 9. Multiple choice (medium)

Which statement about fast (1 to 4 step) image samplers is accurate?

- **A.** All of them are plain GANs trained from scratch
- **B.** SDXL-Turbo uses adversarial diffusion distillation, while Latent Consistency Models use consistency distillation without an adversarial loss
- **C.** They all use flow matching only, with no distillation
- **D.** They can only generate one class

<details>
<summary>Answer</summary>

**B.** SDXL-Turbo uses adversarial diffusion distillation, while Latent Consistency Models use consistency distillation without an adversarial loss

Adversarial losses are one distillation tool, not the only one.

- **A:** They are distilled from diffusion teachers, not trained as standalone GANs.
- **C:** Distillation is how they get to few steps.
- **D:** They are text-conditioned, open-domain models.

</details>

## 10. Multiple choice (hard)

You need diverse, open-domain text-to-image generation with stable training, and latency of a few seconds is fine. Which is the better default?

- **A.** A GAN, because sampling is one forward pass
- **B.** A latent diffusion or flow-matching model, because training is stable and it covers the distribution; distill later if latency becomes a problem
- **C.** A gazetteer of stock images
- **D.** An autoencoder trained with squared error only

<details>
<summary>Answer</summary>

**B.** A latent diffusion or flow-matching model, because training is stable and it covers the distribution; distill later if latency becomes a problem

GANs shine in narrow domains with tight latency. For open-domain quality and coverage, diffusion and flow matching are the default.

- **A:** One-pass speed doesn't fix open-domain coverage and training instability, and latency isn't the constraint here.
- **C:** Retrieval isn't generation.
- **D:** Squared-error decoders produce blurry averages.

</details>
