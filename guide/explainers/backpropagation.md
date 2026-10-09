---
title: How backpropagation assigns blame
summary: Follow one neuron through a forward pass, a loss, and a backward pass with real numbers, and see how the chain rule hands every weight its share of the blame for one bad prediction.
lab: backprop
lesson: backpropagation
minutes: 7
---

# How backpropagation assigns blame

<!-- lab:backprop -->
![Gradient norm reaching the first layer as network depth grows from 1 to 30. With sigmoid activations it collapses toward 1e-12 by depth 20; with ReLU and He initialization it stays roughly level.](../figures/backpropagation.png)
<!-- /lab -->

A network makes a prediction, the prediction is wrong, and now every weight needs to know how much of the mistake was its fault. Backpropagation answers that for all weights at once. We'll follow one neuron with real numbers; on the website the lab beside the text changes as you scroll.

<!-- step:forward -->
## The forward pass

Our neuron has two inputs, $x_1 = 2$ and $x_2 = 1$, two weights, $w_1 = 0.5$ and $w_2 = -1$, and a bias $b = 0$. First it takes a weighted sum:

```math
z = w_1 x_1 + w_2 x_2 + b = 0.5 \times 2 + (-1) \times 1 + 0 = 0
```

Then the sigmoid squashes $z$ into a probability: $\hat{y} = \sigma(0) = 0.5$. The neuron is saying "fifty-fifty". Rows 1 and 2 of the lab show exactly this. Keep these intermediate values in mind: the backward pass will reuse every one of them.

<!-- step:loss -->
## The loss

The true label is $y = 1$, so a guess of 0.5 is only half right. Binary cross-entropy turns that into one number:

```math
L = -\ln \hat{y} = -\ln 0.5 = 0.693
```

Watch $w_1$ swing in the lab. When it grows, $z$ grows, $\hat{y}$ climbs toward 1 and the loss shrinks toward 0. When it falls, $\hat{y}$ drops and the loss climbs fast, because cross-entropy punishes a confident wrong answer hard. Training means nudging the weights so this number goes down. To nudge them we need to know which way each one should move, and by how much.

<!-- step:chain -->
## Passing the gradient backwards

The loss doesn't see $w_1$ directly. It sees $\hat{y}$, which sees $z$, which sees $w_1$. The chain rule says: to get the loss's sensitivity to something far back, multiply the local sensitivities along the way.

So we start at the loss and walk backwards one node at a time. Through the loss node, $\partial L/\partial \hat{y} = (\hat{y} - y)/(\hat{y}(1-\hat{y}))$. Through the sigmoid node, multiply by its slope $\hat{y}(1-\hat{y})$. Those two factors cancel, and what reaches $z$ is strikingly simple:

```math
\frac{\partial L}{\partial z} = \hat{y} - y = 0.5 - 1 = -0.5
```

This is the error signal, row 4 of the lab. Negative means "$z$ should go up".

<!-- step:gradients -->
## A gradient for every weight

One more step back. Since $z = w_1x_1 + w_2x_2 + b$, nudging $w_1$ moves $z$ by $x_1$ times as much. So each weight's gradient is the error signal times the input it multiplies:

```math
\frac{\partial L}{\partial w_1} = -0.5 \times 2 = -1, \quad \frac{\partial L}{\partial w_2} = -0.5 \times 1 = -0.5, \quad \frac{\partial L}{\partial b} = -0.5
```

That is the blame assignment. $w_1$ gets twice the blame of $w_2$ because its input was twice as large: it had twice the say in the mistake. Watch $x_1$ swing in the lab and the $w_1$ gradient follows it. A weight attached to an input of 0 gets no blame at all.

<!-- step:update -->
## The update

Now step every weight against its gradient, scaled by the learning rate $\eta = 0.1$:

```math
w_1 = 0.5 - 0.1(-1) = 0.6, \quad w_2 = -1 - 0.1(-0.5) = -0.95, \quad b = 0 - 0.1(-0.5) = 0.05
```

Run the forward pass again: $z = 0.6 \times 2 - 0.95 + 0.05 = 0.30$, so $\hat{y} = \sigma(0.30) = 0.574$. The prediction moved from 0.5 toward the label 1, and the loss fell. In the lab, watch $\eta$ grow: the gradients stay the same, but the step gets longer and the new prediction lands further along. The gradient picks the direction; $\eta$ decides how far to trust it.

<!-- step:flip -->
## Flip the label

Set $y = 0$ and run it again. The forward pass doesn't change at all: $z = 0$, $\hat{y} = 0.5$, $L = 0.693$, because the loss for a 0.5 guess is the same either way.

But the error signal becomes $0.5 - 0 = +0.5$, and since every gradient is that signal times an input, all of them flip sign: $+1$, $+0.5$, $+0.5$. The update now pushes $w_1$ down to 0.4, and the new prediction drops to 0.426, toward the new label. Same arithmetic, opposite blame. The sign of $\hat{y} - y$ alone decides which way every weight moves.

<!-- step:cost -->
## Why it's cheap

Look back at what we computed. One subtraction, $\hat{y} - y$, fed all three gradients, and each needed only one extra multiply by a value the forward pass had already stored.

In a deep network the same thing happens layer by layer. The gradient arriving at a layer is computed once and reused by everything upstream of it, so the backward pass does roughly the same kind of work as the forward pass, about one to two times its cost. Compare that with nudging each weight separately and re-running the network: for a billion weights, that's two billion forward passes. Backpropagation gets every gradient for the price of about one.

**Try it yourself:** edit any input or weight in the lab and all nine steps recompute, or open the [full lesson](../lessons/11-gradient-descent-backprop/02-backpropagation.md) for vanishing gradients, a finite-difference check, and the PyTorch version.
