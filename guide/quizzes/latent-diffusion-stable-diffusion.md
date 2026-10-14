<!-- GENERATED from latent-diffusion-stable-diffusion.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Latent diffusion and Stable Diffusion

Covers the lesson [Latent diffusion and Stable Diffusion](../lessons/13-deep-architectures/05-latent-diffusion-stable-diffusion.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/latent-diffusion-stable-diffusion/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

SD 1.5 encodes a $512 \times 512 \times 3$ image into a $64 \times 64 \times 4$ latent. How many times fewer values does the diffusion model process?

<details>
<summary>Answer</summary>

**48**

$786{,}432 / 16{,}384 = 48$. Spatial positions drop by $8 \times 8 = 64$; channels go from 3 to 4.

</details>

## 2. Calculation (medium)

Using $\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$ with $\epsilon_\varnothing = -0.4$, $\epsilon_c = 0.1$, and $w = 5$, what is $\tilde\epsilon$?

<details>
<summary>Answer</summary>

**2.1** (within ±0.001)

$-0.4 + 5 \times (0.1 - (-0.4)) = -0.4 + 5 \times 0.5 = -0.4 + 2.5 = 2.1$.

</details>

## 3. Calculation (medium)

A new autoencoder's latents have standard deviation 4.0 on your dataset. What scaling factor should multiply the latents so diffusion sees unit variance?

<details>
<summary>Answer</summary>

**0.25** (within ±0.001)

$s_f = 1 / 4.0 = 0.25$. SD 1.5 uses 0.18215 for the same reason (its latents have standard deviation of roughly 5.5).

</details>

## 4. Multiple choice (easy)

How are the two stages of latent diffusion trained?

- **A.** Jointly, end to end, with one loss
- **B.** The autoencoder is trained first and frozen; then the diffusion model is trained on its latents
- **C.** The diffusion model is trained first, then the autoencoder learns to decode its samples
- **D.** Only the decoder is trained; the encoder is random

<details>
<summary>Answer</summary>

**B.** The autoencoder is trained first and frozen; then the diffusion model is trained on its latents

Stage 1 learns a compact, decodable latent space. Stage 2 trains the usual noise-prediction loss on $z = s_f E(x)$.

- **A:** Joint training would let the latent space drift while diffusion is trying to model it.
- **C:** The diffusion model needs latents to train on, so the autoencoder must come first.
- **D:** A random encoder would give latents unrelated to image content.

</details>

## 5. Multiple choice (medium)

In Stable Diffusion's cross-attention layers, what plays the role of queries, keys, and values?

- **A.** Queries, keys, and values all come from the text tokens
- **B.** Queries come from image (latent) features; keys and values come from the text-encoder tokens
- **C.** Queries come from text tokens; keys and values from image features
- **D.** Queries, keys, and values all come from image features

<details>
<summary>Answer</summary>

**B.** Queries come from image (latent) features; keys and values come from the text-encoder tokens

Each latent position asks which prompt tokens are relevant and takes a weighted mix of their values. The attention matrix is (positions) by (text tokens).

- **A:** Then the output would describe text only, not each image position.
- **C:** That would produce one output per text token, not per image position.
- **D:** That's self-attention, which also appears in the denoiser, but it doesn't bring in the prompt.

</details>

## 6. Multiple choice (medium)

A paper writes guidance as $(1 + w')\epsilon_c - w'\epsilon_\varnothing$ and recommends $w' = 3$. What `guidance_scale` gives the same result in diffusers, which uses $\epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$?

- **A.** 2
- **B.** 3
- **C.** 4
- **D.** 6

<details>
<summary>Answer</summary>

**C.** 4

Expanding the paper's form gives $\epsilon_\varnothing + (1 + w')(\epsilon_c - \epsilon_\varnothing)$, so $w = 1 + w' = 4$.

- **A:** That subtracts instead of adding 1.
- **B:** That ignores the offset between the two conventions.
- **D:** That doubles instead of adding 1.

</details>

## 7. Multiple choice (hard)

After fine-tuning SD 1.5 on your own images, samples come out washed out and low contrast, although loss looked normal. What is the most likely cause?

- **A.** The learning rate was too low
- **B.** The training latents were not multiplied by the VAE scaling factor (0.18215), so the denoiser learned the wrong variance
- **C.** The text encoder was frozen
- **D.** Too many sampling steps

<details>
<summary>Answer</summary>

**B.** The training latents were not multiplied by the VAE scaling factor (0.18215), so the denoiser learned the wrong variance

The schedule assumes unit-variance data; unscaled latents have standard deviation of about 5.5. Encoding and decoding must both apply the factor.

- **A:** A low learning rate gives a model that barely changes, not systematically washed-out colors.
- **C:** Freezing the text encoder is the normal setup.
- **D:** More steps generally helps or does nothing; it doesn't wash out colors.

</details>

## 8. Multiple choice (hard)

A team trains a class-conditional latent diffusion model but never drops the condition. At sampling they apply classifier-free guidance with a null token. What happens?

- **A.** Guidance works as usual
- **B.** The null-token prediction is untrained, so the guided direction is meaningless and samples degrade
- **C.** The model becomes unconditional
- **D.** Sampling is faster

<details>
<summary>Answer</summary>

**B.** The null-token prediction is untrained, so the guided direction is meaningless and samples degrade

CFG needs the unconditional prediction, which comes from training with condition dropout (about 10%).

- **A:** Guidance relies on a trained unconditional branch.
- **C:** The conditional predictions are still fine; only the guidance combination is broken.
- **D:** Guidance still costs two network passes per step.

</details>

## 9. Multiple choice (medium)

Which statement about SD3 and Flux.1 compared with SD 1.5 is accurate?

- **A.** They are pixel-space diffusion models
- **B.** They use transformer denoisers, 16-channel latents, a T5 text encoder alongside CLIP, and rectified-flow training
- **C.** They use the same autoencoder as SD 1.5, so SD 1.5 LoRAs work on them
- **D.** They removed text conditioning

<details>
<summary>Answer</summary>

**B.** They use transformer denoisers, 16-channel latents, a T5 text encoder alongside CLIP, and rectified-flow training

Same two-stage recipe, upgraded parts.

- **A:** Both are latent models.
- **C:** Their autoencoders and latent spaces differ, so components don't transfer.
- **D:** They are text-to-image models.

</details>

## 10. Select all that apply (medium)

Which changes reduce text-to-image latency for a latent diffusion model? Select all that apply.

- **A.** Distill to a 1 to 4 step sampler
- **B.** Batch the conditional and unconditional passes together
- **C.** Use fp16 or bf16 weights and fast attention kernels
- **D.** Increase guidance scale from 5 to 12

<details>
<summary>Answer</summary>

**A, B, C**

Fewer steps, batched guidance, and cheaper arithmetic all cut time. Guidance scale changes the output, not the cost.

</details>
