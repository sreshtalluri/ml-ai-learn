---
title: How self-attention mixes words
summary: Follow one word through self-attention, from its vector to queries and keys, dot-product scores, scaling, softmax weights and a blend of values, and see how that lets "it" find the noun it refers to.
lab: attention
lesson: self-attention
minutes: 7
---

# How self-attention mixes words

<!-- lab:attention -->
![Left: a 6 by 6 attention heatmap for "the cat sat on the mat" where every token attends to every other. Middle: the same with a causal mask, so the upper triangle is zero and the first token attends only to itself. Right: for random 512-dimensional vectors, softmax of unscaled dot products puts nearly all weight on one key, while scaling by the square root of d_k spreads it out.](../figures/self-attention.png)
<!-- /lab -->

In "The animal didn't cross the street because it was too tired," you know "it" means the animal, not the street. Self-attention is how a transformer makes that connection: every word builds its new meaning by borrowing from the words that matter to it. This explainer follows one word through the whole calculation; on the website the lab beside the text changes as you scroll.

<!-- step:embed -->
## Words become vectors

A model can't compare words, only numbers. So each token is looked up in an embedding table and becomes a vector. In the lab, "the cat sat on the mat" turns into six rows of four numbers each, a matrix $X$ of shape $[6, 4]$. A position signal is added to each row, because attention on its own has no idea which word came first.

The orange outline marks "cat". We'll follow that one row through every step. The embeddings here are synthetic, seeded random numbers, so the patterns show the mechanics rather than real meaning. A trained model learns them.

<!-- step:qkv -->
## Three views of every word: query, key, value

Each row is multiplied by three different weight matrices to give three new vectors:

```math
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
```

The **query** is what this word is looking for. The **key** is what it offers to anyone searching. The **value** is the information it hands over if chosen. A pronoun's query might say "I need a singular, animate noun"; a noun's key might say "I am one."

In the lab each is two numbers wide, so $d_k = 2$ and $Q$, $K$, $V$ all have shape $[6, 2]$. The weights are fixed random numbers here. Training is what makes them meaningful.

<!-- step:scores -->
## Every query meets every key

To ask how well word $i$'s question matches word $j$'s answer, take a dot product: $S_{ij} = q_i \cdot k_j$. It's large when the two vectors point the same way, near zero when they're unrelated, and negative when they point apart. Doing this for every pair at once is one matrix product, $S = QK^\top$, with shape $[6, 6]$: one row per asking word, one column per word being asked.

With small numbers you can check by hand. Take $q = [1, 0]$, $k_1 = [1, 0]$ and $k_2 = [0, 1]$. Then $q \cdot k_1 = 1 \times 1 + 0 \times 0 = 1$ and $q \cdot k_2 = 0$. The first key is the better match. Watch the outline walk down the rows: every word scores every other word.

<!-- step:scale -->
## Turn the volume down: divide by √d_k

Before going further, every score is divided by $\sqrt{d_k}$. In the example, $\sqrt{2} = 1.4142$, so the scores $[1, 0]$ become $[0.7071, 0]$.

Why bother? A dot product adds up $d_k$ products. If each has variance 1, the sum has standard deviation $\sqrt{d_k}$. The lesson's script measures 7.98 for $d_k = 64$ and 22.42 for $d_k = 512$. Feed scores that spread out into softmax and one word grabs nearly all the weight, while the gradients that should train $W_Q$ and $W_K$ shrink to almost nothing. Dividing by $\sqrt{d_k}$ brings the spread back to about 1. The ranking doesn't change, only how sharp the result is.

<!-- step:softmax -->
## Scores become shares

Softmax turns each row of scores into weights that are positive and sum to 1:

```math
A_{ij} = \frac{\exp(S'_{ij})}{\sum_l \exp(S'_{il})}
```

For the example, $e^{0.7071} = 2.0281$ and $e^0 = 1$, which sum to $3.0281$. The weights are $[2.0281/3.0281,\ 1/3.0281] = [0.6698,\ 0.3302]$. Each word hands out exactly one unit of attention, split according to how well each key matched.

In the lab this is the blue heatmap: darker cells are where a word looks hardest. The tour runs without the causal mask, so every word can see every other. A decoder adds the mask, which zeroes everything to the right of the diagonal so no word can peek at the future.

<!-- step:output -->
## The output is a blend of values

Now each word collects its new vector: the values of all words, mixed in proportion to the weights.

```math
\text{output}_i = \sum_j A_{ij}\, v_j
```

With $v_1 = [10, 0]$ and $v_2 = [0, 6]$:

```math
0.6698\,[10, 0] + 0.3302\,[0, 6] = [6.698,\ 1.981]
```

The output is mostly $v_1$, because that key matched best, but $v_2$ still adds a third. That's why it's called soft attention: nothing is picked outright, everything is blended. In the lab, the new row for "cat" carries a bit of every word in the sentence, in exactly the proportions of its heatmap row. Shapes: $[6, 6] \times [6, 2] = [6, 2]$, one new vector per word.

<!-- step:referent -->
## How "it" finds "animal"

Put it together. When the model processes "it," its query is compared with the key of every earlier word. Training has shaped $W_Q$ and $W_K$ so a pronoun's query points the same way as the keys of nouns that could be its referent: animate, singular, nearby. "animal" scores high, "street" scores low, softmax turns that gap into most of the weight, and the new vector for "it" is mostly the value of "animal."

The lab's course example is exactly this in miniature: the query matches $k_1$, so $k_1$ gets weight 0.67 and the output is mostly $v_1$. A real model repeats this across many heads and layers, and each head can learn a different kind of lookup.

Try it yourself: work through the full [self-attention lesson](../lessons/14-transformers/01-self-attention.md), with every shape, the causal mask and multi-head attention.
