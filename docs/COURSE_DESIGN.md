# Course design review

A pass over the whole curriculum by the standard a course designer would apply: for each lesson, what learners most often get wrong, and what they should be able to *manipulate* to fix it. This document decides which interactive labs get built, in what order, and how lessons use them.

## Principles

1. **Predict, then check.** A lab that only shows an animation teaches little. Each lab in a lesson comes with two or three "Try it" prompts that ask the learner to predict an outcome before touching a control. Being surprised is where the learning happens.
2. **One misconception per lab.** Every lab targets a specific wrong belief (for example "more trees always help," "a cluster is a category," "the model will refuse injected instructions"). If we can't name the misconception, the lab is decoration.
3. **Every number visible.** Labs show the arithmetic, not just the picture: the Gini before and after a split, the chunk scores in a retrieval, the residual each boosting tree fits.
4. **Same data as the lesson.** Labs start on the lesson's worked example, so the hand calculation and the interactive version agree to the digit.
5. **Static fallback for GitHub readers.** Every lab sits on top of a figure produced by a Python script, so the GitHub edition loses interactivity, not content.

## Tools considered

| Tool | Verdict |
|---|---|
| In-repo lab framework (`LabFrame`, `Plot`, SVG in React) | **Used for all interactive labs.** Small bundles, full control over every number, works in light and dark themes, tested with Vitest. |
| `dataviz` skill | **Used as the chart standard.** Its validator caught that the dark-mode chart palette was too light; the palette was re-stepped until all six checks passed (light and dark). Its rules (one axis, thin marks, legends for 2+ series, text never in series color) apply to every lab. |
| `diagram` skill (Mermaid → Excalidraw/SVG) | Good for static architecture diagrams in the GitHub edition. Not interactive, so the production-architecture lesson uses a custom clickable diagram instead. |
| "Archify" | Not installed. The closest plugin, `archlens`, diagrams codebases rather than concepts. |
| Chart libraries (D3, Recharts, Plotly) | Not needed: every lab is a few hundred lines of SVG over shared primitives, and adding a library would only grow the bundle. |

## Lesson-by-lesson review

Priority: **A** = build now (high learning leverage, misconception is common and costly), **B** = next, **C** = a static figure is enough.

| Module · lesson | Misconception to fix | What the learner manipulates | Lab | Priority | Status |
|---|---|---|---|---|---|
| 0 · Vectors | "Dot product is just arithmetic" | drag two vectors, watch projection and cosine | vector playground | B | backlog |
| 0 · Calculus | "Gradient is a number" | move a point on a surface, see the gradient arrow | covered by gradient-descent lab | C | — |
| 0 · Probability | Ignoring base rates | prevalence, sensitivity, specificity → posterior | Bayes base-rate lab | B | backlog |
| 1 · Paradigms | Picking an algorithm before the feedback signal | answer questions, load scenarios | paradigm-guide | A | done |
| 2 · Workflow | "Random split is always fine" | split type, leakage toggle → reported vs real score | data-split lab | B | backlog |
| 2 · Overfitting | "Lower training error = better model" | polynomial degree, training size, noise → train/val curves | **fit-explorer** | A | **built** |
| 3 · Linear regression | Residuals, gradients, L1 vs L2 | drag points, step gradient descent, penalties | linear-regression | A | done |
| 4 · Metrics | "Accuracy is enough," "0.5 is the threshold" | threshold, error costs, calibration | threshold | A | done |
| 5 · KNN | Scaling doesn't matter | K, metric, scaling | knn | A | done |
| 5 · Naive Bayes | Independence makes probabilities trustworthy | word toggles → posterior | NB calculator | C | Math Lab covers the arithmetic |
| 6 · Decision trees | How a split is actually chosen | pick feature and threshold, see Gini before/after, grow the tree | **decision-tree** | A | **built** |
| 6 · Boosting | "More trees always help," what a tree fits | learning rate, rounds → residuals and test error | **boosting** | A | **built** |
| 7 · K-means | Clusters are facts | place centroids, step, change K and data | kmeans | A | done |
| 7 · DBSCAN/GMM | One method fits all shapes | ε, min_samples on moons and rings | dbscan | B | backlog |
| 8 · PCA | "PC1 is a feature" | rotate a projection axis, see captured variance | **pca** | A | **built** |
| 9 · TF-IDF | Rare vs common words | edit corpus | tfidf | A | done |
| 9 · Embeddings | Words as points | 2D semantic space and analogies | embeddings | B | backlog |
| 10 · Forward pass | What a layer does to numbers | weights, activations, sizes | nn-forward | A | done |
| 11 · Gradient descent | Learning-rate limits, optimizers | lr, optimizer, surface | gradient-descent | A | done |
| 11 · Backprop | Chain rule as bookkeeping | edit inputs, see nine steps | backprop | A | done |
| 12 · Training | Early stopping, dropout | regularization strength, epochs | training curves | B | backlog |
| 13 · CNNs | What a kernel computes | pick kernel, step the window, pool | **convolution** | A | **built** |
| 13 · RNNs | Gradients vanish over time | recurrent weight, steps back | RNN unroll | B | backlog |
| 13 · Diffusion | Noise schedule | timestep slider | diffusion steps | C | figure is enough |
| 14 · Self-attention | Q, K, V and shapes | sentence, mask, step through | attention | A | done |
| 14 · Transformer | Where parameters and compute live | width, layers, context → params, KV cache | transformer sizing | B | backlog |
| 15 · Tokenization | "Tokens are words" | type text, step BPE merges, watch context fill | **tokenizer** | A | **built** |
| 15 · Decoding | Temperature changes truth | T, top-k, top-p, sample | decoding | A | done |
| 15 · Adapting LLMs | RAG vs fine-tuning | requirement → method | method chooser | C | table is enough |
| 16 · RAG | "Bad answers are prompt problems" | chunk size, overlap, k, query → scores, evidence, miss | **rag** | A | **built** |
| 17 · Evaluation | Small eval gains are real | cases fixed/broken, sample size → interval | eval comparison | B | backlog |
| 18 · Architecture | The model is the system | click components, trace a request | **architecture** | A | **built** |
| 18 · Reliability | Retries always help | failure rate, retries, breaker | reliability simulator | B | backlog |
| 19 · Security | The model will refuse injections | toggle controls, run attacks on mocked tools | **prompt-injection** | A | **built** |
| 20 · Plan / projects | — | — | — | C | — |

## Changes to how the course works through

- **Learning paths.** Added [`guide/learning-paths.md`](../guide/learning-paths.md) with three routes: the full course, an AI-engineering fast track for experienced software engineers, and an interview-prep pass. Each lists lessons and labs in order with time estimates.
- **"Try it" prompts.** Every lesson with a lab now has two or three predict-then-check experiments right under the figure, written so they work with the static figure too.
- **Labs everywhere they matter.** The nine labs marked "built" above bring the total to 20, covering every module where a learner can usefully move a parameter.

## Remaining backlog

The B-priority rows above. Each follows the same recipe: a component in `web/src/components/labs/`, an entry in `registry.tsx` and `src/lib/labs.ts`, and a marker around the lesson figure.
