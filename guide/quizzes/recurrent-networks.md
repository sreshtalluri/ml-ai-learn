<!-- GENERATED from recurrent-networks.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Recurrent networks, LSTMs, and GRUs

Covers the lesson [Recurrent networks, LSTMs, and GRUs](../lessons/13-deep-architectures/02-recurrent-networks.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/recurrent-networks/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

Scalar RNN with $w_x = 1$, $w_h = 0.5$, $b = 0$, $h_{t-1} = 0.4$, $x_t = 0.3$. What is the pre-activation $w_x x_t + w_h h_{t-1}$?

<details>
<summary>Answer</summary>

**0.5** (within ±0.0001)

$0.3 + 0.5 \times 0.4 = 0.5$; then $h_t = \tanh(0.5) = 0.462$.

</details>

## 2. Calculation (medium)

In a linear RNN with recurrent weight 0.5, what factor scales the gradient flowing back 10 steps? (4 decimals)

<details>
<summary>Answer</summary>

**0.001** (within ±0.0001)

$0.5^{10} = 0.000977 \approx 0.0010$.

</details>

## 3. Multiple choice (medium)

Which LSTM component lets information and gradients flow across many steps with little change?

- **A.** The output gate alone
- **B.** The cell state, updated additively and scaled by the forget gate
- **C.** The input embedding
- **D.** The softmax layer

<details>
<summary>Answer</summary>

**B.** The cell state, updated additively and scaled by the forget gate

$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$; with $f_t \approx 1$ the cell state passes through almost unchanged.

</details>

## 4. Multiple choice (medium)

Why can't a vanilla RNN be parallelized across time steps during training?

- **A.** It has too many parameters.
- **B.** Step t needs the hidden state from step t − 1, so steps must be computed in order.
- **C.** GPUs don't support recurrence.
- **D.** It uses softmax.

<details>
<summary>Answer</summary>

**B.** Step t needs the hidden state from step t − 1, so steps must be computed in order.

Transformers compute all positions at once because attention doesn't depend on a previous step's output during training.

</details>

## 5. Multiple choice (easy)

Training an RNN, the loss suddenly becomes NaN. What is the standard first fix?

- **A.** Increase the learning rate.
- **B.** Gradient clipping (and possibly a lower learning rate).
- **C.** Remove the hidden state.
- **D.** Use a larger batch of zeros.

<details>
<summary>Answer</summary>

**B.** Gradient clipping (and possibly a lower learning rate).

Exploding gradients through time are common in RNNs; clipping caps the gradient norm.

</details>

## 6. Reflection (hard)

When might you still choose a GRU over a transformer today?

<details>
<summary>Answer</summary>

**Model answer.** For streaming or on-device settings: a GRU processes one step at a time with constant memory and tiny compute, which suits real-time sensor or audio streams and low-power hardware. A transformer's cost grows with context length and usually needs more data. For small time-series problems a GRU (or gradient boosting on lag features) can be a simpler, adequate baseline.

The trade-off is per-step cost and memory versus parallel training and long-range modeling.

</details>
