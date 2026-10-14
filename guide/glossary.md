# Glossary

Plain-English definitions first, then formal ones, with a small example and the lesson that teaches each term. On the [website](https://sreshtalluri.github.io/ml-ai-learn/glossary/) this page is searchable, and linked terms show their definition on hover inside lessons.

### Activation

**Plain English:** The nonlinear function a neuron applies to its weighted sum, so the network can bend its decision boundaries.

**Formal:** $a = f(z)$ with $z = w\cdot x + b$; common $f$ are ReLU, GELU, sigmoid, tanh.

**Example:** ReLU turns $z = -1.5$ into 0 and $z = 2$ into 2.

**Related:** [Logit](#logit), [Softmax](#softmax)

**Lesson:** [The forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md)

### Advantage

**Plain English:** How much better an action turned out than what was expected from that state.

**Formal:** $A(s, a) = Q(s, a) - V(s)$; estimated as $G_t - V(s_t)$ (REINFORCE with a baseline), with GAE (PPO), or as a group-normalized reward (GRPO).

**Example:** An episode returns $-2.71$ from a cell whose baseline is $-3.5$, so the advantage is $+0.79$, and the actions taken become more likely.

**Related:** [Policy gradient](#policy-gradient), [Generalized advantage estimation](#generalized-advantage-estimation)

**Lesson:** [Policy gradients and REINFORCE](lessons/26-reinforcement-learning/04-policy-gradients-reinforce.md)

### Agent

**Plain English:** A system that uses a model in a loop to plan, call tools, observe results, and decide what to do next.

**Formal:** A controller that alternates model calls and tool executions until a stop condition, under step, cost, and permission limits.

**Example:** A support agent looks up an order, checks the refund policy, and drafts a reply for human approval.

**Related:** [Tool calling](#tool-calling), [Prompt injection](#prompt-injection)

**Lesson:** [Production architecture](lessons/20-production-ai/01-production-architecture.md)

### Agent memory

**Plain English:** How an agent keeps information beyond one model call: a small working set in the prompt, plus an external store it writes to and searches with tools.

**Formal:** Two tiers: a bounded main context (system prompt, editable core memory, recent messages) and an external store (archival notes, evicted messages), connected by memory tools whose results are appended to the next prompt.

**Example:** The agent saves "dog is named Miso" with `archival_memory_insert`; 30 turns later, after the message was evicted, `archival_memory_search("dog")` pages it back in.

**Related:** [Context window](#context-window), [RAG](#rag), [Tool calling](#tool-calling)

**Lesson:** [Agent memory](lessons/19-agents/03-agent-memory.md)

### All-gather

**Plain English:** Every GPU shares its slice so all GPUs end with the full concatenated tensor.

**Formal:** Each rank contributes one chunk and receives the other $N - 1$; it sends $\frac{N-1}{N}T$ bytes per rank. The second half of a ring all-reduce.

**Example:** FSDP all-gathers a layer's sharded weights just before computing that layer.

**Related:** [Reduce-scatter](#reduce-scatter), [All-reduce](#all-reduce), [FSDP](#fsdp)

**Lesson:** [Collective operations from scratch](lessons/27-training-at-scale/03-collective-ops-from-scratch.md)

### All-reduce

**Plain English:** A group operation that leaves every GPU holding the sum (or average) of everyone's tensors.

**Formal:** A ring all-reduce is a reduce-scatter followed by an all-gather, $2(N-1)$ steps in total, and sends $2\frac{N-1}{N}T$ bytes per rank for a $T$-byte tensor.

**Example:** Data-parallel training all-reduces gradients every step: 13.48 GB of bf16 gradients over 8 GPUs means 23.58 GB sent per GPU.

**Related:** [Reduce-scatter](#reduce-scatter), [All-gather](#all-gather), [Data parallelism](#data-parallelism)

**Lesson:** [Collective operations from scratch](lessons/27-training-at-scale/03-collective-ops-from-scratch.md)

### Anomaly detection

**Plain English:** Finding data points that don't fit the normal pattern, usually without labeled examples of the anomalies.

**Formal:** Learn a score $a(x)$ from mostly normal data so that unusual $x$ get high scores; flag $a(x) > \tau$, with $\tau$ set from an alert budget.

**Example:** Flagging the top 200 of 1 million daily transactions by isolation forest score for analyst review.

**Related:** [Isolation forest](#isolation-forest), [Robust z-score](#robust-z-score)

**Lesson:** [Anomaly detection](lessons/24-applied-ml/05-anomaly-detection.md)

### Approximate nearest neighbor search

**Plain English:** Finding vectors that are probably the closest to a query, checking only a small part of the collection instead of every vector.

**Formal:** An index returns a set $S$ of $k$ vectors that approximates the exact top-$k$ under a distance; quality is measured as recall@$k = |S \cap \text{exact top-}k|/k$ against brute force.

**Example:** An IVF index with 1,024 cells that probes 8 scans about 0.8% of a million vectors.

**Related:** [HNSW](#hnsw), [Embedding](#embedding), [Cosine similarity](#cosine-similarity)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Attention

**Plain English:** A learned, soft lookup: each token gathers information from other tokens, weighted by how relevant they are.

**Formal:** $\text{softmax}(QK^\top/\sqrt{d_k})V$, where queries, keys, and values are linear projections of the token vectors.

**Example:** With $q = [1, 0]$, keys $[1,0]$ and $[0,1]$, the weights are $[0.67, 0.33]$.

**Related:** [Transformer](#transformer), [Context window](#context-window)

**Lesson:** [Self-attention](lessons/14-transformers/01-self-attention.md)

### Autocorrelation

**Plain English:** How strongly a series is correlated with a shifted copy of itself.

**Formal:** $r_k = \sum_t (y_t - \bar y)(y_{t-k} - \bar y) / \sum_t (y_t - \bar y)^2$; plotting $r_k$ against $k$ gives the ACF.

**Example:** Daily data with a weekly pattern has ACF spikes at lags 7, 14, and 21.

**Related:** [Stationarity](#stationarity)

**Lesson:** [Time series](lessons/24-applied-ml/04-time-series.md)

### Autograd

**Plain English:** A framework feature that records the operations applied to tensors and computes gradients by running the chain rule backwards over that record.

**Formal:** Reverse-mode automatic differentiation: one backward pass gives $\partial L / \partial \theta$ for every parameter $\theta$ that has `requires_grad=True`. Gradients add into `.grad`, so they must be zeroed each step.

**Example:** For $L = (wx + b - y)^2$ with $x = 2, w = 3, b = 1, y = 10$, `L.backward()` sets `w.grad = -12` and `b.grad = -6`.

**Related:** [Backpropagation](#backpropagation), [Gradient](#gradient)

**Lesson:** [Intro to PyTorch](lessons/10-neural-networks/03-intro-to-pytorch.md)

### Automatic differentiation

**Plain English:** Computing exact gradients by recording each operation in the forward pass and applying the chain rule to that record in reverse.

**Formal:** Reverse-mode autodiff visits the computation graph in reverse topological order; each node adds (local derivative × upstream gradient) to its inputs' `.grad`.

**Example:** For `a * a + a` at a = 3, the two paths into `a` contribute 6 and 1, so `a.grad` = 7.

**Related:** [Backpropagation](#backpropagation)

**Lesson:** [Backpropagation](lessons/11-gradient-descent-backprop/02-backpropagation.md)

### Average precision

**Plain English:** A single-number summary of the precision-recall curve: the average precision at the rank of each positive example.

**Formal:** $\text{AP} = \frac{1}{P}\sum_k \text{precision@}k \cdot \mathbb{1}[\text{item } k \text{ is positive}]$, where P is the number of positives.

**Example:** Positives at ranks 1, 2, 4, 6, 9 give AP = (1 + 1 + 0.75 + 0.667 + 0.556)/5 = 0.794.

**Lesson:** [Classification metrics](lessons/04-classification/02-classification-metrics.md)

### Backpropagation

**Plain English:** The method for computing how much each weight contributed to the loss, by applying the chain rule backward through the network.

**Formal:** Reverse-mode automatic differentiation over the computational graph, multiplying local derivatives from the loss to each parameter.

**Example:** For a sigmoid neuron with cross-entropy, $\partial L/\partial w_j = (\hat{y} - y)x_j$.

**Related:** [Gradient](#gradient), [Gradient descent](#gradient-descent)

**Lesson:** [Backpropagation](lessons/11-gradient-descent-backprop/02-backpropagation.md)

### Baseline

**Plain English:** The simplest reasonable approach, used to judge whether a model is actually useful.

**Formal:** A reference score from a trivial predictor (mean, majority class, last value) or a simple model on the same evaluation data.

**Example:** With 5% positives, "always negative" scores 95% accuracy, so a 95% model has learned nothing.

**Related:** [Metric](#metric)

**Lesson:** [Workflow, splits, and leakage](lessons/02-ml-workflow/01-ml-workflow.md)

### Batch

**Plain English:** A small group of training examples processed together in one optimization step.

**Formal:** A mini-batch $B$ used to estimate the gradient $\frac{1}{|B|}\sum_{i\in B}\nabla\ell_i$.

**Example:** 1,024 examples with batch size 32 give 32 steps per epoch.

**Related:** [Epoch](#epoch), [Gradient descent](#gradient-descent)

**Lesson:** [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md)

### Bayesian optimization

**Plain English:** A search that fits a cheap model of score versus settings, with uncertainty, and uses it to choose the next setting to try.

**Formal:** A surrogate (Gaussian process or TPE) gives $\mu(\lambda), \sigma(\lambda)$; an acquisition such as expected improvement $(\mu - f^* - \xi)\Phi(z) + \sigma\phi(z)$ picks the next $\lambda$.

**Example:** Optuna's default TPE sampler is a Bayesian optimization method.

**Related:** [Random search](#random-search), [Hyperparameter](#hyperparameter)

**Lesson:** [Hyperparameter tuning](lessons/24-applied-ml/03-hyperparameter-tuning.md)

### Bellman equation

**Plain English:** The value of where you are equals the reward for one step plus the discounted value of where you land.

**Formal:** $V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)[r + \gamma V^\pi(s')]$; the optimality version replaces the average over actions with a max.

**Example:** Value iteration repeats the optimality version until values stop changing.

**Related:** [Markov decision process](#markov-decision-process), [Q-learning](#q-learning), [Temporal-difference learning](#temporal-difference-learning)

**Lesson:** [MDPs, states, actions, and rewards](lessons/26-reinforcement-learning/01-mdps-states-actions-rewards.md)

### Benjamini-Hochberg

**Plain English:** A correction for testing many metrics at once that controls the expected fraction of false "wins" rather than the chance of any false win.

**Formal:** Sort p-values p_(1) ≤ … ≤ p_(m); find the largest r with p_(r) ≤ q·r/m; reject hypotheses 1..r. Controls false discovery rate at q (for independent or positively dependent tests).

**Example:** For p = 0.001, 0.008, 0.012, 0.041, … with m = 10 and q = 0.05, the cut falls at rank 3 (0.012 ≤ 0.015, 0.041 > 0.02).

**Related:** [p-value](#p-value)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Bias-variance trade-off

**Plain English:** Simple models are consistently wrong (bias); flexible models change a lot with the training data (variance); the best model balances both.

**Formal:** Expected squared error $= \text{bias}^2 + \text{variance} + \sigma^2$.

**Example:** A degree-1 polynomial underfits a sine curve; degree 15 overfits it.

**Related:** [Overfitting](#overfitting), [Regularization](#regularization)

**Lesson:** [Overfitting and bias-variance](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md)

### BIO tagging

**Plain English:** A way to label each token so a classifier can mark entity spans: B starts an entity, I continues it, O means outside any entity.

**Formal:** For a span $(s, e, X)$, token $s$ gets `B-X`, tokens $s+1..e-1$ get `I-X`, all others `O`; `I-X` may only follow `B-X` or `I-X`.

**Example:** "New York City mayor" is tagged `B-LOC I-LOC I-LOC O`.

**Related:** [Named entity recognition](#named-entity-recognition), [Viterbi algorithm](#viterbi-algorithm)

**Lesson:** [Named entity recognition](lessons/09-classical-nlp/03-named-entity-recognition.md)

### BM25

**Plain English:** A keyword-ranking formula that scores documents by how often they contain the query's words, giving more weight to rare words and less to very long documents.

**Formal:** $\text{BM25}(q, D) = \sum_{t \in q} \text{IDF}(t)\,\frac{f(t,D)(k_1+1)}{f(t,D) + k_1(1 - b + b\,|D|/\text{avgdl})}$, typically $k_1 \approx 1.2$, $b \approx 0.75$.

**Example:** A query for an error code like "E1234" is matched exactly by BM25 even when a dense embedding treats it as noise.

**Related:** [TF-IDF](#tf-idf), [Reciprocal rank fusion](#reciprocal-rank-fusion)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Bootstrap

**Plain English:** Estimating the uncertainty of a statistic by resampling your data with replacement many times and recomputing it each time.

**Formal:** Draw B resamples of size n with replacement, compute the statistic on each, and take the 2.5th and 97.5th percentiles as a 95% interval.

**Example:** 870 of 1,000 correct gives a bootstrap 95% interval of about [0.850, 0.890] for accuracy.

**Lesson:** [Probability and statistics](lessons/00-foundations/03-probability-and-statistics.md)

### Bootstrap sample

**Plain English:** A resample of the training set, the same size as the original, drawn with replacement, so some rows repeat and others are left out.

**Formal:** Draw $n$ indices uniformly from $\{1,\dots,n\}$ with replacement; each row is left out with probability $(1 - 1/n)^n \approx e^{-1} \approx 0.368$.

**Example:** From 1,000 rows, a bootstrap sample contains about 632 distinct rows; the roughly 368 left out are that tree's out-of-bag rows.

**Related:** [Out-of-bag error](#out-of-bag-error)

**Lesson:** [Random forests and gradient boosting](lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md)

### Broadcasting

**Plain English:** NumPy's rule for combining arrays of different shapes by stretching size-1 dimensions without copying data.

**Formal:** Align shapes from the right; each pair of dimensions must be equal or one must be 1 (missing dimensions count as 1). The result takes the larger size in each position.

**Example:** `[1000, 3] - [3]` gives `[1000, 3]`; `[1000] - [1000, 1]` gives `[1000, 1000]`, usually a bug.

**Related:** [Data leakage](#data-leakage)

**Lesson:** [Python toolkit](lessons/00-foundations/04-python-toolkit.md)

### Calibration

**Plain English:** Whether a model's predicted probabilities match how often things actually happen.

**Formal:** A model is calibrated if $P(y = 1 \mid \hat{p} = p) = p$ for all $p$; measured with reliability diagrams or the Brier score.

**Example:** Of all emails scored 0.8 spam, about 80% should really be spam.

**Related:** [Logit](#logit), [Hallucination](#hallucination)

**Lesson:** [Classification metrics](lessons/04-classification/02-classification-metrics.md)

### Canary rollout

**Plain English:** Shift a growing share of traffic to a new version and roll back automatically if guard metrics get worse.

**Formal:** Stages s₁ < … < 1; promote while metric_canary ≤ k · metric_baseline for every gate.

**Lesson:** [Production architecture](lessons/20-production-ai/01-production-architecture.md)

### Catastrophic forgetting

**Plain English:** When training a model on a new, narrow task makes it lose skills it had before.

**Formal:** Performance on the original distribution drops after fine-tuning because gradient updates for the new objective overwrite weights the old behavior depended on.

**Example:** A model fine-tuned for many epochs on support tickets gets better at tickets but worse at general reasoning and safety refusals.

**Related:** [Fine-tuning](#fine-tuning), [LoRA](#lora)

**Lesson:** [Fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md)

### Chain-of-thought

**Plain English:** Prompting a model to write out intermediate reasoning steps before its final answer, so each step is a small prediction.

**Formal:** If each step fails independently with probability $\varepsilon$, an $s$-step answer is right with probability $(1-\varepsilon)^s$; writing steps as tokens lowers $\varepsilon$ because later tokens can condition on earlier results.

**Example:** At $\varepsilon = 0.15$ a 4-step answer is right with probability $0.85^4 = 0.522$; at $\varepsilon = 0.05$ it is $0.95^4 = 0.815$.

**Related:** [Few-shot prompting](#few-shot-prompting), [Self-consistency](#self-consistency), [ReAct](#react)

**Lesson:** [Few-shot and chain-of-thought](lessons/28-prompting-and-context/02-few-shot-cot.md)

### Checkpoint

**Plain English:** A saved snapshot of a model's weights (and often optimizer state) during or after training.

**Formal:** A serialized copy of $\theta$ at step $t$, used to resume training, evaluate, or deploy.

**Example:** Early stopping deploys the checkpoint with the lowest validation loss, not the last one.

**Related:** [Epoch](#epoch)

**Lesson:** [Training and regularization](lessons/12-training-regularization/01-training-and-regularization.md)

### Chinchilla-optimal

**Plain English:** The split between model size and training tokens that gives the lowest loss for a fixed training compute budget, roughly 20 tokens per parameter.

**Formal:** $N^*, D^* = \arg\min_{6ND = C} L(N, D)$. With Approach 3 constants, $N^* \propto C^{0.45}$ and $D^* \propto C^{0.55}$, so the ratio drifts with budget.

**Example:** For $C = 6 \times 10^{22}$ FLOPs at 20 tokens per parameter: $N = 22.4$B, $D = 447$B tokens. Models served heavily are trained far past this ratio on purpose.

**Related:** [Scaling law](#scaling-law)

**Lesson:** [Scaling laws](lessons/15-llms/06-scaling-laws.md)

### Chunking

**Plain English:** Splitting documents into smaller pieces that can be retrieved individually.

**Formal:** Partitioning text into segments of size $S$ with overlap $O$, giving about $\lceil (N - O)/(S - O)\rceil$ chunks.

**Example:** A 1,000-word document with 200-word chunks and 50-word overlap gives 7 chunks.

**Related:** [RAG](#rag), [Embedding](#embedding)

**Lesson:** [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md)

### Class weights

**Plain English:** Multipliers on the loss that make mistakes on a rare class cost more, so the model pays attention to it.

**Formal:** Balanced weights are $w_c = n / (C \cdot n_c)$, so each class carries total weight $n / C$. For logistic regression they inflate the predicted odds by about $n_0 / n_1$, which mainly moves the decision threshold.

**Example:** With 2,279 negatives and 121 positives, $w_0 = 0.5265$ and $w_1 = 9.917$: one positive counts as 18.83 negatives.

**Related:** [SMOTE](#smote), [Calibration](#calibration), [Loss](#loss)

**Lesson:** [Imbalanced data](lessons/24-applied-ml/02-imbalanced-data.md)

### Classifier-free guidance

**Plain English:** A sampling trick that strengthens how much a diffusion model follows its prompt by extrapolating from an unconditional prediction toward the conditional one.

**Formal:** $\tilde\epsilon = \epsilon_\theta(z_t, t, \varnothing) + w[\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing)]$; requires dropping the condition (about 10%) during training. $w = 1$ is plain conditional sampling.

**Example:** $\epsilon_\varnothing = 0.2$, $\epsilon_c = 0.5$, $w = 3$ gives $0.2 + 3 \times 0.3 = 1.1$.

**Related:** [Latent diffusion](#latent-diffusion)

**Lesson:** [Latent diffusion and Stable Diffusion](lessons/13-deep-architectures/05-latent-diffusion-stable-diffusion.md)

### Cold start

**Plain English:** The problem of making good predictions for a new user or item that has no interaction history yet.

**Formal:** For an entity with no logged interactions, collaborative signals are undefined, so the model must rely on content and context features, priors such as popularity, or deliberate exploration.

**Example:** A video uploaded a minute ago has no views; the item tower embeds its title and frames so it can still be retrieved, and re-ranking gives it a small freshness boost.

**Related:** [Two-tower model](#two-tower-model), [Position bias](#position-bias)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

### Concept drift

**Plain English:** The world's answer to the same input changes, so a model that was right becomes wrong even though its inputs look normal.

**Formal:** A change in $P(y \mid x)$. Contrast covariate drift (change in $P(x)$) and prior drift (change in $P(y)$). Only outcome monitoring with labels can detect it.

**Example:** Fraudsters change tactics, so the same basket size now carries a different fraud risk.

**Related:** [Drift](#drift), [Population Stability Index](#population-stability-index)

**Lesson:** [Model monitoring and drift](lessons/22-ml-system-design/03-model-monitoring-and-drift.md)

### Constrained decoding

**Plain English:** Forcing a model's output to follow a format, such as a JSON schema, by blocking any next token that would break it.

**Formal:** At each step, tokens outside the allowed set $A_t$ (from a grammar compiled from the schema) get probability 0 and the rest are renormalized: $\tilde p(y_t) \propto p(y_t)\,\mathbb{1}[y_t \in A_t]$.

**Example:** A tool call generated under the schema always parses, though `flight_id` may still be the wrong flight.

**Related:** [Tool calling](#tool-calling), [Temperature](#temperature)

**Lesson:** [Tool use and agents](lessons/19-agents/01-tool-use-and-agents.md)

### Context engineering

**Plain English:** Deciding everything that goes into a model call's context window (instructions, tools, retrieved documents, history, memory) and in what order, under a token budget.

**Formal:** Choose and order components so that input tokens plus the output reserve fit the window $W$, while maximizing task quality; with full history, total billed input over $T$ calls is $TF + h\,T(T-1)/2$.

**Example:** Selecting 10 of 30 tools, 3 of 8 documents, and compacting history cut a synthetic 20-call conversation from 231,500 to 93,750 input tokens.

**Related:** [Context window](#context-window), [Lost in the middle](#lost-in-the-middle), [Prefix caching](#prefix-caching), [RAG](#rag)

**Lesson:** [Context engineering](lessons/28-prompting-and-context/03-context-engineering.md)

### Context window

**Plain English:** The maximum number of tokens a language model can consider at once, prompt and output combined.

**Formal:** The maximum sequence length $n_{\max}$ over which attention is computed.

**Example:** Anything beyond the window is invisible to the model unless retrieved and inserted.

**Related:** [Token](#token), [Attention](#attention)

**Lesson:** [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md)

### Continuous batching

**Plain English:** A serving scheduler that adds and removes requests from the running batch at every decode step, instead of waiting for a whole batch to finish.

**Formal:** Admission is decided per iteration: when a sequence emits its end token, its slot (and KV memory) is given to a waiting request on the next step. Also called in-flight or iteration-level batching.

**Example:** Two slots, requests of 2, 8, 3, and 3 tokens: static batching finishes at step 11 (1.45 tokens/step), continuous batching at step 8 (2.0 tokens/step).

**Related:** [Batch](#batch), [Paged attention](#paged-attention), [KV cache](#kv-cache)

**Lesson:** [Serving LLMs](lessons/18-llm-inference/02-serving-llms.md)

### Coordinate descent

**Plain English:** An optimizer that updates one parameter at a time, solving for it exactly while holding the others fixed, and sweeps until nothing moves.

**Formal:** For $j = 1, \dots, d$: $w_j \leftarrow \arg\min_{w_j} L(w_1, \dots, w_d)$; repeat. Lasso solvers use it because each one-weight problem has a closed form (soft-thresholding).

**Example:** scikit-learn's `Lasso` and `ElasticNet` are fitted by coordinate descent.

**Related:** [Soft-thresholding](#soft-thresholding)

**Lesson:** [Regularization and regression metrics](lessons/03-regression/02-regularization-and-regression-metrics.md)

### Cosine similarity

**Plain English:** How closely two vectors point in the same direction, ignoring their lengths.

**Formal:** $\cos(a, b) = \frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$, between −1 and 1.

**Example:** $[1, 2, 0]$ and $[2, 1, 1]$ have cosine 0.730.

**Related:** [Embedding](#embedding), [TF-IDF](#tf-idf)

**Lesson:** [From text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md)

### Cost governor

**Plain English:** Runtime budget caps (per call, steps, per task, spend rate, monthly) that halt an agent before it crosses a limit.

**Lesson:** [Building reliable agents](lessons/19-agents/02-reliable-agents.md)

### Cross-entropy

**Plain English:** A loss that punishes a model for giving low probability to the correct answer.

**Formal:** $-\ln p_{\text{true}}$ for one example; binary form $-[y\ln p + (1-y)\ln(1-p)]$.

**Example:** Probability 0.9 on the true class costs 0.105; probability 0.01 costs 4.6.

**Related:** [Loss](#loss), [Perplexity](#perplexity)

**Lesson:** [Logistic regression](lessons/04-classification/01-logistic-regression.md)

### CTC

**Plain English:** Connectionist temporal classification: a way to train a model to output text from many audio frames without knowing which frame produced which letter.

**Formal:** The model emits a distribution over tokens plus a blank at each frame; a path collapses to text by merging repeats and removing blanks, and $P(y \mid x)$ sums over all paths that collapse to $y$, computed with a forward algorithm in $O(T \cdot (2U+1))$.

**Example:** With frame probabilities $(0.4, 0.6), (0.5, 0.5), (0.7, 0.3)$ for (blank, a), $P(\text{"a"}) = 0.77$ and the loss is $-\ln 0.77 = 0.261$.

**Related:** [Word error rate](#word-error-rate), [Spectrogram](#spectrogram)

**Lesson:** [Speech recognition: CTC, transducers, and attention](lessons/29-speech-and-audio/02-speech-recognition-asr.md)

### CUPED

**Plain English:** A variance-reduction trick for A/B tests that subtracts the part of each user's outcome that was already predictable from their behavior before the experiment.

**Formal:** $Y^{\text{cuped}} = Y - \theta(X - \bar{X})$ with $\theta = \operatorname{Cov}(X, Y)/\operatorname{Var}(X)$, where $X$ is a pre-experiment covariate; the variance shrinks by a factor $1 - \rho^2$ and the estimate stays unbiased.

**Example:** With $\rho = 0.5$ between last month's and this month's sessions, variance drops 25%, so a test needing 14,749 users per arm needs about 11,062.

**Related:** [Statistical power](#statistical-power), [Minimum detectable effect](#minimum-detectable-effect)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Curse of dimensionality

**Plain English:** In many dimensions, random points are all about equally far apart, so "nearest neighbor" carries little information and data needed to cover the space grows exponentially.

**Example:** For 1,000 uniform random points and a random query, nearest/farthest distance ratio was 0.007 in 2 dimensions and 0.898 in 1,000 dimensions.

**Lesson:** [K-nearest neighbors](lessons/05-instance-and-probabilistic/01-k-nearest-neighbors.md)

### Data leakage

**Plain English:** When training uses information that won't be available at prediction time, making results look better than they are.

**Formal:** Any dependence of training features or preprocessing on test data, future data, or the target itself.

**Example:** Fitting a scaler on all data before splitting, or including "refund issued" when predicting fraud.

**Related:** [Baseline](#baseline), [Generalization](#generalization)

**Lesson:** [Workflow, splits, and leakage](lessons/02-ml-workflow/01-ml-workflow.md)

### Data parallelism

**Plain English:** Every GPU holds a full copy of the model and trains on a different slice of the batch; gradients are averaged so all copies stay identical.

**Formal:** With equal shards, the mean of shard gradients equals the full-batch gradient; one gradient all-reduce per step. Throughput scales with GPUs, memory per GPU does not drop. PyTorch's DistributedDataParallel (DDP).

**Example:** A 7B-style model still needs about 108.6 GB per GPU under DDP, however many GPUs you add.

**Related:** [All-reduce](#all-reduce), [ZeRO](#zero), [Tensor parallelism](#tensor-parallelism)

**Lesson:** [Scaling and distributed training](lessons/27-training-at-scale/04-scaling-distributed.md)

### Dead ReLU

**Plain English:** A ReLU unit whose input is negative for every example, so it always outputs 0 and gets zero gradient, and stops learning.

**Formal:** Unit $j$ is dead on a dataset if $z_j(x) \le 0$ for all $x$; then $\partial L / \partial z_j = 0$.

**Example:** With hidden biases initialized to −3, 97% of a second layer's units were dead and its gradients were about 1,000 times smaller than in a healthy network.

**Related:** [Activation](#activation), [Gradient](#gradient)

**Lesson:** [Debugging neural networks](lessons/12-training-regularization/02-debugging-neural-networks.md)

### Deadly triad

**Plain English:** Function approximation, bootstrapping, and off-policy learning together can make value learning diverge.

**Formal:** Term from Sutton and Barto: any two of the three are stable in general, but combining all three removes convergence guarantees for TD methods.

**Example:** DQN has all three; replay and a target network make it work in practice without restoring the guarantee.

**Related:** [Deep Q-network (DQN)](#deep-q-network-dqn), [Target network](#target-network)

**Lesson:** [Deep Q-networks (DQN)](lessons/26-reinforcement-learning/03-dqn.md)

### Deep Q-network (DQN)

**Plain English:** Q-learning where a neural network, instead of a table, predicts the value of each action in a state.

**Formal:** Minimizes $\mathbb{E}\big[\ell\big(Q(s,a;\theta) - (r + \gamma(1-d)\max_{a'} Q(s',a';\theta^-))\big)\big]$ over minibatches from a replay buffer, with a target network $\theta^-$ and usually the Huber loss $\ell$.

**Example:** A 16 → 32 → 4 network learns the 6-step shortest path on a 4×4 GridWorld; the 2015 version learned dozens of Atari games from pixels.

**Related:** [Experience replay](#experience-replay), [Target network](#target-network), [Double DQN](#double-dqn)

**Lesson:** [Deep Q-networks (DQN)](lessons/26-reinforcement-learning/03-dqn.md)

### Demographic parity

**Plain English:** A fairness criterion that requires every group to be selected at the same rate.

**Formal:** $P(\hat Y = 1 \mid A = a)$ is equal for all groups $a$. With unequal base rates it usually conflicts with equal error rates and equal precision.

**Example:** Approving 31% of group A and 52% of group B violates demographic parity even if both groups have the same TPR and FPR.

**Related:** [Equalized odds](#equalized-odds)

**Lesson:** [Fairness](lessons/21-safety-security/02-fairness.md)

### Dice coefficient

**Plain English:** An overlap score for masks, equal to F1 computed on pixels; used as a loss for imbalanced segmentation.

**Formal:** $\text{Dice} = 2|P \cap Y| / (|P| + |Y|) = 2\,\text{IoU}/(1 + \text{IoU})$; soft Dice loss replaces the sets with predicted probabilities.

**Example:** 40 shared pixels, 60 predicted, 50 true: Dice = 80/110 = 0.727, IoU = 0.571.

**Related:** [Intersection over union](#intersection-over-union), [Loss](#loss)

**Lesson:** [Semantic segmentation and U-Net](lessons/25-computer-vision/02-semantic-segmentation-unet.md)

### Differential privacy

**Plain English:** A guarantee that a computation's output would look almost the same whether or not any one person's record was in the data.

**Formal:** $M$ is $(\varepsilon, \delta)$-DP if $P[M(D) \in S] \le e^\varepsilon P[M(D') \in S] + \delta$ for all neighbouring datasets $D, D'$ and output sets $S$. Costs add up over queries; post-processing is free.

**Example:** Releasing a count with Laplace noise of scale $1/\varepsilon$ is $\varepsilon$-DP; at $\varepsilon = 0.5$ the typical error is 2.

**Related:** [DP-SGD](#dp-sgd)

**Lesson:** [Differential privacy](lessons/21-safety-security/03-differential-privacy.md)

### Discount factor

**Plain English:** How much a reward later counts compared with the same reward now.

**Formal:** $\gamma \in [0, 1]$; the return is $G_t = \sum_k \gamma^k r_{t+k}$, and the effective horizon is about $1/(1-\gamma)$ steps.

**Example:** With $\gamma = 0.9$, a reward 10 steps away keeps $0.9^{10} \approx 0.35$ of its weight.

**Related:** [Markov decision process](#markov-decision-process), [Bellman equation](#bellman-equation)

**Lesson:** [MDPs, states, actions, and rewards](lessons/26-reinforcement-learning/01-mdps-states-actions-rewards.md)

### Distillation

**Plain English:** Training a small model to imitate a large one, to get similar quality more cheaply.

**Formal:** Minimizing the divergence between the student's outputs and the teacher's (soft) outputs, often alongside the true labels.

**Example:** A small classifier trained on a large LLM's labels serves high-volume traffic at a fraction of the cost.

**Related:** [Quantization](#quantization), [Inference](#inference)

**Lesson:** [Reliability, cost, and observability](lessons/20-production-ai/02-reliability-cost-and-observability.md)

### Double DQN

**Plain English:** A DQN variant that reduces overoptimistic Q-values by letting one network choose the next action and another score it.

**Formal:** $y = r + \gamma(1-d)\,Q\big(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\big)$.

**Example:** Online values $[-5.5, -5.1, -5.7, -4.95]$ pick action 4; the target network scores it $-5.0$, so $y = -1 + 0.99 \times (-5.0) = -5.95$ instead of the vanilla $-5.851$.

**Related:** [Deep Q-network (DQN)](#deep-q-network-dqn)

**Lesson:** [Deep Q-networks (DQN)](lessons/26-reinforcement-learning/03-dqn.md)

### DP-SGD

**Plain English:** Training with differential privacy: limit how much each example can push the model, then add noise to every update.

**Formal:** Clip each per-example gradient to norm $C$, add $\mathcal{N}(0, \sigma^2 C^2 I)$ to their sum, divide by the batch size, and step; an accountant tracks the total $\varepsilon$ across steps (Abadi et al. 2016).

**Example:** 100 full-batch steps at $\sigma = 4$, $\delta = 10^{-5}$ give $\varepsilon \approx 15$; $\sigma = 16$ gives about 3.2.

**Related:** [Differential privacy](#differential-privacy), [Gradient descent](#gradient-descent)

**Lesson:** [Differential privacy](lessons/21-safety-security/03-differential-privacy.md)

### DPO

**Plain English:** Direct preference optimization: training a model on pairs of better and worse answers so it prefers the better kind, without a separate reward model or reinforcement learning loop.

**Formal:** $\mathcal{L} = -\log\sigma\big(\beta[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}]\big)$ with a frozen reference model $\pi_{\text{ref}}$.

**Example:** At the start of training the policy equals the reference, so the loss is $\ln 2 = 0.693$.

**Related:** [Fine-tuning](#fine-tuning), [LoRA](#lora)

**Lesson:** [Fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md)

### Drift

**Plain English:** When the data or the world changes after deployment, so a model's performance degrades.

**Formal:** Data drift is a change in $P(x)$; concept drift is a change in $P(y \mid x)$; prompt drift is a change in what users ask.

**Example:** A demand model trained before a price change starts under-predicting.

**Related:** [Generalization](#generalization), [Calibration](#calibration)

**Lesson:** [Reliability, cost, and observability](lessons/20-production-ai/02-reliability-cost-and-observability.md)

### Embedding

**Plain English:** A dense vector that represents an item (word, token, document, user) so that similar items are close together.

**Formal:** A learned mapping $e: \text{item} \to \mathbb{R}^d$, often a row of an embedding matrix of shape $[V, d]$.

**Example:** "cat" and "kitten" have high cosine similarity even though the words differ.

**Related:** [Cosine similarity](#cosine-similarity), [Token](#token)

**Lesson:** [Word embeddings](lessons/09-classical-nlp/02-word-embeddings.md)

### Entropy

**Plain English:** The average surprise of a probability distribution: how uncertain you are before seeing the outcome.

**Formal:** $H(P) = -\sum_x p(x)\log p(x)$. Bits with $\log_2$, nats with $\ln$.

**Example:** A fair coin has 1 bit of entropy; a coin that lands heads 99% of the time has 0.08 bits.

**Related:** [Cross-entropy](#cross-entropy), [KL divergence](#kl-divergence), [Perplexity](#perplexity)

**Lesson:** [Information theory](lessons/00-foundations/05-information-theory.md)

### Epoch

**Plain English:** One full pass over the training data.

**Formal:** $\lceil n/|B|\rceil$ optimization steps for $n$ examples and batch size $|B|$.

**Example:** Validation loss is usually checked once per epoch for early stopping.

**Related:** [Batch](#batch), [Checkpoint](#checkpoint)

**Lesson:** [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md)

### Epsilon-greedy

**Plain English:** Usually take the best-looking action, but with a small probability take a random one, so the agent keeps discovering.

**Formal:** With probability $\varepsilon$ choose uniformly among actions; otherwise choose $\arg\max_a Q(s, a)$.

**Example:** With $\varepsilon = 0.1$ and 4 actions, the greedy action is taken with probability $0.9 + 0.1/4 = 0.925$.

**Related:** [Q-learning](#q-learning), [SARSA](#sarsa)

**Lesson:** [Q-learning and SARSA](lessons/26-reinforcement-learning/02-q-learning-sarsa.md)

### Equalized odds

**Plain English:** A fairness criterion that requires the model to make the same kinds of mistakes in every group: equal true positive rates and equal false positive rates.

**Formal:** $P(\hat Y = 1 \mid Y = y, A = a)$ is equal across groups for $y \in \{0, 1\}$. Equal opportunity is the $y = 1$ half. When base rates differ, it cannot hold together with equal precision (unless the classifier is perfect).

**Example:** TPR 0.8 and FPR 0.1 in both groups satisfies equalized odds.

**Related:** [Demographic parity](#demographic-parity), [Calibration](#calibration)

**Lesson:** [Fairness](lessons/21-safety-security/02-fairness.md)

### Evaluator-optimizer

**Plain English:** A workflow where one call drafts an answer and another critiques it against criteria, looping until it passes or a cap is hit.

**Lesson:** [Building reliable agents](lessons/19-agents/02-reliable-agents.md)

### Expectation-maximization (EM)

**Plain English:** A way to fit models with hidden labels, such as which Gaussian generated each point: guess soft labels, refit the parameters to them, and repeat. The likelihood never decreases.

**Formal:** E-step: $\gamma_{ik} = \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) / \sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)$. M-step: $N_k = \sum_i \gamma_{ik}$, $\pi_k = N_k/n$, $\mu_k = \sum_i \gamma_{ik} x_i / N_k$, $\Sigma_k$ the $\gamma$-weighted covariance.

**Example:** a point with responsibilities $(0.24, 0.76)$ contributes 0.24 of itself to component 1's mean and 0.76 to component 2's.

**Lesson:** [DBSCAN, hierarchical clustering, and Gaussian mixtures](lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md)

### Experience replay

**Plain English:** A buffer of recent transitions that an RL agent trains on in random minibatches instead of in the order they happened.

**Formal:** Store $(s, a, r, s', d)$ tuples in a fixed-capacity FIFO buffer $\mathcal{D}$ and sample minibatches uniformly (or by priority) for each gradient step. Valid for off-policy methods such as Q-learning.

**Example:** With capacity 2,000 and batch 32, each transition is reused in many updates, and a batch mixes transitions from many episodes.

**Related:** [Deep Q-network (DQN)](#deep-q-network-dqn), [Target network](#target-network)

**Lesson:** [Deep Q-networks (DQN)](lessons/26-reinforcement-learning/03-dqn.md)

### Expert capacity

**Plain English:** The fixed number of token slots each expert gets per batch; extra tokens sent to a full expert are dropped.

**Formal:** $C = \lceil c \cdot Tk/E \rceil$ for $T$ tokens, top-$k$ routing, $E$ experts, and capacity factor $c$.

**Example:** 4,096 tokens, 64 experts, top-2, $c = 1.25$: $C = \lceil 1.25 \times 128 \rceil = 160$.

**Related:** [Mixture of experts](#mixture-of-experts)

**Lesson:** [Mixture of experts](lessons/15-llms/05-mixture-of-experts.md)

### Faithfulness

**Plain English:** How much of an answer is actually supported by the sources it was given.

**Formal:** The fraction of the answer's claims that an entailment model or judge marks as supported by the retrieved context.

**Example:** Two of three claims supported gives faithfulness 0.67, even if the third claim happens to be true.

**Related:** [Calibration](#calibration)

**Lesson:** [Evaluating LLM systems](lessons/17-llm-evaluation/01-llm-evaluation.md)

### Feature

**Plain English:** An input variable the model uses to make a prediction.

**Formal:** A component $x_j$ of the input vector $x \in \mathbb{R}^p$.

**Example:** Square footage when predicting house prices.

**Related:** [Label](#label), [Parameter](#parameter)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Feature store

**Plain English:** A system that defines each feature once and serves the same values both to training pipelines and to the live model.

**Formal:** An offline store of historical feature values supporting point-in-time (as-of) joins for training, plus a low-latency online store for serving, both fed by the same feature definitions.

**Example:** "Clicks on this item in the last hour" is computed by one streaming job, written to the online store for serving and to the offline store so training examples get the value as of each impression.

**Related:** [Training-serving skew](#training-serving-skew), [Data leakage](#data-leakage)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

### Few-shot prompting

**Plain English:** Putting a few input-output examples in the prompt so the model copies the task's format and style.

**Formal:** In-context learning: the examples are part of the conditioning context $p(y \mid \text{examples}, x)$; no weights change.

**Example:** One worked example ending in `Answer: 20` made a model end every answer with `Answer: N`, so a strict parser found the answer in 100% of outputs instead of 24%.

**Related:** [Chain-of-thought](#chain-of-thought), [Fine-tuning](#fine-tuning)

**Lesson:** [Few-shot and chain-of-thought](lessons/28-prompting-and-context/02-few-shot-cot.md)

### Fine-tuning

**Plain English:** Continuing to train a pretrained model on task-specific data so it changes its behavior.

**Formal:** Further optimization of some or all pretrained weights (or added adapters such as LoRA) on a new dataset.

**Example:** Fine-tuning a model on support transcripts to adopt a consistent tone and format.

**Related:** [RAG](#rag), [LoRA](#lora)

**Lesson:** [Adapting LLMs](lessons/15-llms/03-adapting-llms.md)

### FlashAttention

**Plain English:** An exact way to compute attention that keeps small tiles of the work in fast on-chip memory, so the full attention matrix is never written to slow GPU memory.

**Formal:** Tiles $Q$, $K$, $V$ into SRAM blocks and uses an online softmax (running max and sum) to produce the same output as standard attention, with $O(T)$ extra memory instead of $O(T^2)$ and far less memory traffic.

**Example:** At 8,192 tokens, one head's 16-bit score matrix would be 134 MB; FlashAttention never materializes it.

**Related:** [Attention](#attention), [KV cache](#kv-cache)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Flow matching

**Plain English:** Training a generator by teaching a network the velocity that moves points along straight paths between noise and data, then following those velocities from noise.

**Formal:** $x_t = (1-t)x_0 + t x_1$, loss $\mathbb{E}\lVert v_\theta(x_t, t) - (x_1 - x_0)\rVert^2$; sample by integrating $dx/dt = v_\theta$ from $t = 1$ to $0$. Rectified flow adds reflow: retraining on the model's own (noise, sample) pairs to straighten paths.

**Example:** $x_0 = 2$, $x_1 = -0.5$, $t = 0.25$ gives $x_t = 1.375$ and target velocity $-2.5$.

**Related:** [Latent diffusion](#latent-diffusion)

**Lesson:** [Flow matching and rectified flows](lessons/13-deep-architectures/06-flow-matching-rectified-flows.md)

### Focal loss

**Plain English:** Cross-entropy that down-weights examples the model already gets right, so rare hard examples dominate the gradient.

**Formal:** $\text{FL} = -\alpha(1 - p_t)^\gamma \ln p_t$, where $p_t$ is the probability of the true class; $\gamma = 0$ gives cross-entropy.

**Example:** With $\gamma = 2$, an easy example at $p_t = 0.9$ keeps 1% of its cross-entropy; a hard one at $p_t = 0.1$ keeps 81%.

**Related:** [Cross-entropy](#cross-entropy), [Loss](#loss)

**Lesson:** [Loss functions](lessons/10-neural-networks/02-loss-functions.md)

### FSDP

**Plain English:** PyTorch's built-in version of fully sharded data parallelism: each GPU keeps only a slice of the model and gathers each layer's weights when it needs them.

**Formal:** Fully Sharded Data Parallel, equivalent to ZeRO stage 3. Weights are all-gathered before each wrapped module's forward and backward, and gradients are reduce-scattered. `fully_shard` in FSDP2; `SHARD_GRAD_OP` and `HYBRID_SHARD` give ZeRO-2-like and per-node variants.

**Example:** A 70B-style model first fits on 80 GB GPUs at 16 GPUs (about 72 GB each).

**Related:** [ZeRO](#zero), [All-gather](#all-gather)

**Lesson:** [ZeRO and FSDP](lessons/27-training-at-scale/05-zero-parameter-sharding.md)

### GAN

**Plain English:** A generative adversarial network: a generator makes fake samples and a discriminator learns to tell them from real ones; the generator improves by fooling it.

**Formal:** $\min_G \max_D \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]$; the generator usually minimizes the non-saturating loss $-\log D(G(z))$.

**Example:** For fixed $G$, the best discriminator is $D^*(x) = p_{\text{data}}/(p_{\text{data}} + p_g)$, which is 0.5 everywhere at equilibrium.

**Related:** [Mode collapse](#mode-collapse), [Loss](#loss)

**Lesson:** [GANs, generator versus discriminator](lessons/13-deep-architectures/04-gans-generator-discriminator.md)

### Generalization

**Plain English:** How well a model performs on new data it never saw during training.

**Formal:** Expected loss on the data distribution, estimated by held-out data that played no part in building the model.

**Example:** A model with 99% training accuracy and 70% test accuracy generalizes poorly.

**Related:** [Overfitting](#overfitting), [Data leakage](#data-leakage)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Generalized advantage estimation

**Plain English:** An advantage estimate that blends one-step and multi-step lookaheads, trading bias from the critic against variance from the real returns.

**Formal:** $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, $\hat A_t = \sum_{k \ge 0} (\gamma\lambda)^k \delta_{t+k}$. $\lambda = 0$ gives the one-step TD error; $\lambda = 1$ gives the Monte Carlo return minus $V$.

**Example:** With $\gamma = 0.99$, $\lambda = 0.95$, TD errors $-0.091$ and then $-0.1$ at the end give $\hat A = -0.091 + 0.9405 \times (-0.1) = -0.185$.

**Related:** [Advantage](#advantage), [PPO](#ppo)

**Lesson:** [PPO](lessons/26-reinforcement-learning/05-ppo.md)

### Goodput

**Plain English:** The share of requests that met every latency target at once.

**Formal:** Fraction of requests with TTFT ≤ a, TPOT ≤ b, and end-to-end latency ≤ c.

**Lesson:** [Serving LLMs](lessons/18-llm-inference/02-serving-llms.md)

### Gradient

**Plain English:** The direction and steepness of the loss surface: which way each parameter should move to increase the loss fastest.

**Formal:** $\nabla_\theta L = [\partial L/\partial\theta_1, \dots, \partial L/\partial\theta_P]$.

**Example:** For $L = x^2 + 2y^2$ at $(1, 1)$, $\nabla L = (2, 4)$.

**Related:** [Gradient descent](#gradient-descent), [Backpropagation](#backpropagation)

**Lesson:** [Calculus for ML](lessons/00-foundations/02-calculus-for-ml.md)

### Gradient accumulation

**Plain English:** Running several small batches and adding up their gradients before taking one optimizer step, to get the effect of a batch too big for memory.

**Formal:** Effective batch size = micro-batch size × accumulation steps. With a masked token loss, sum the per-token gradients and divide by the total token count across micro-batches, not the average of per-micro-batch means.

**Example:** Micro-batch 4 with 8 accumulation steps trains like a batch of 32.

**Related:** [Calibration](#calibration)

**Lesson:** [Fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md)

### Gradient check

**Plain English:** Testing a hand-written backward pass by comparing its gradients with finite-difference estimates.

**Formal:** $g_{\text{num}} = [L(w + h) - L(w - h)]/2h$; relative error below about $10^{-7}$ (float64) passes, around $10^{-3}$ or more indicates a bug.

**Example:** A backward pass missing the ReLU mask gives relative error about 1; the correct one gives $2 \times 10^{-7}$.

**Related:** [Backpropagation](#backpropagation), [Autograd](#autograd)

**Lesson:** [Numerical stability](lessons/00-foundations/06-numerical-stability.md)

### Gradient checkpointing

**Plain English:** Saving memory in training by keeping only some layer inputs and recomputing the rest of the forward pass during backward.

**Formal:** Store the input of every $k$-th layer; during backward re-run each segment's forward to rebuild its activations. Gradients are identical; compute rises by about 33% (one extra forward per step). Also called activation recomputation.

**Example:** A 7B-style model at 2,048 tokens keeps about 9.13 GB of activations, or 0.822 GB checkpointing every layer.

**Related:** [Activation](#activation), [Backpropagation](#backpropagation)

**Lesson:** [Gradient checkpointing](lessons/27-training-at-scale/02-gradient-checkpointing.md)

### Gradient clipping

**Plain English:** Capping how big one update can be, so a single bad batch cannot throw the weights far off.

**Formal:** Global-norm clipping rescales all gradients by $\min(1, c / \lVert g \rVert)$, where $\lVert g \rVert$ is the L2 norm over every parameter; the update keeps its direction.

**Example:** Gradients $[3, 4]$ and $[12]$ have norm 13; clipping at 1 multiplies all of them by $1/13$.

**Related:** [Mixed precision](#mixed-precision), [Gradient](#gradient)

**Lesson:** [Gradient clipping and mixed precision](lessons/27-training-at-scale/01-gradient-clipping-amp.md)

### Gradient descent

**Plain English:** Repeatedly nudging parameters a small step downhill on the loss surface.

**Formal:** $\theta \leftarrow \theta - \eta\nabla_\theta L$; stable along curvature $\lambda$ only if $\eta < 2/\lambda$.

**Example:** With $\eta = 0.1$, $w = 1$ and gradient $-7.33$, the new weight is $1.733$.

**Related:** [Learning rate](#learning-rate), [Gradient](#gradient)

**Lesson:** [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md)

### Grounding

**Plain English:** Tying a model's answer to supplied evidence, so every claim can be traced to a source.

**Formal:** The fraction of an answer's claims supported by the provided context, often with citations.

**Example:** "Refunds are allowed within 30 days [2]," where source 2 contains that sentence.

**Related:** [RAG](#rag), [Hallucination](#hallucination)

**Lesson:** [Evaluating LLM systems](lessons/17-llm-evaluation/01-llm-evaluation.md)

### Grouped-query attention

**Plain English:** Attention where several query heads share one key/value head, which shrinks the KV cache with little quality loss.

**Formal:** With $n_h$ query heads and $n_{kv}$ KV heads ($1 < n_{kv} < n_h$), each group of $n_h / n_{kv}$ query heads attends using the same $K$ and $V$. $n_{kv} = n_h$ is multi-head attention (MHA); $n_{kv} = 1$ is multi-query attention (MQA).

**Example:** 32 query heads and 8 KV heads cut the KV cache to a quarter of MHA's: 131,072 instead of 524,288 bytes per token in a synthetic 8B-style model.

**Related:** [KV cache](#kv-cache), [Attention](#attention)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### GRPO

**Plain English:** Group relative policy optimization: RL for LLMs that scores several sampled answers per prompt and pushes up the ones better than the group average, without a value model.

**Formal:** For $G$ responses with rewards $r_i$, every token of response $i$ gets advantage $(r_i - \text{mean}(r)) / \text{std}(r)$; the loss is PPO's clipped surrogate plus a KL penalty to a reference model.

**Example:** Rewards $[1, 0, 0, 1]$ have mean 0.5 and std 0.5, giving advantages $[1, -1, -1, 1]$.

**Related:** [PPO](#ppo), [Policy gradient](#policy-gradient), [DPO](#dpo)

**Lesson:** [Fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md)

### Hallucination

**Plain English:** A fluent, confident statement that isn't supported by facts or the provided sources.

**Formal:** Generated content inconsistent with the source context or the world, arising because the model optimizes likely text rather than truth.

**Example:** Inventing a refund policy detail that appears in no document.

**Related:** [Grounding](#grounding), [Calibration](#calibration)

**Lesson:** [Adapting LLMs](lessons/15-llms/03-adapting-llms.md)

### Hinge loss

**Plain English:** The SVM's per-point penalty: zero for points safely past the margin, growing linearly for points inside it or on the wrong side.

**Formal:** $\max(0,\, 1 - y\,f(x))$ with labels $y \in \{-1, +1\}$.

**Example:** $y = +1$, $f(x) = -0.2$ gives loss $1.2$; $f(x) = 3$ gives 0.

**Lesson:** [Support vector machines](lessons/05-instance-and-probabilistic/03-support-vector-machines.md)

### HNSW

**Plain English:** Hierarchical navigable small world: a vector index that links each vector to its near neighbors in a layered graph and searches by hopping toward the query.

**Formal:** Each node keeps up to $M$ links per upper layer and $2M$ at layer 0; search descends greedily from the top layer, then runs best-first search with a candidate list of size `efSearch`.

**Example:** With $M = 32$ and 4-byte IDs, the bottom-layer links cost about 256 bytes per vector, 256 MB per million vectors.

**Related:** [Approximate nearest neighbor search](#approximate-nearest-neighbor-search), [Embedding](#embedding)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Huber loss

**Plain English:** A regression loss that is squared for small errors and linear for large ones, so outliers can't dominate.

**Formal:** $\tfrac12 r^2$ if $|r| \le \delta$, else $\delta(|r| - \tfrac12\delta)$, with residual $r$.

**Example:** Residuals $[-0.5, 0.5, 0, 10]$ give mean Huber 2.44 ($\delta = 1$) versus MSE 25.1.

**Related:** [Loss](#loss), [Overfitting](#overfitting)

**Lesson:** [Loss functions](lessons/10-neural-networks/02-loss-functions.md)

### Hyperparameter

**Plain English:** A setting you choose before training that controls how learning happens.

**Formal:** A value fixed outside the optimization of $\theta$, usually selected on a validation set.

**Example:** Learning rate, tree depth, $K$ in K-nearest neighbors.

**Related:** [Parameter](#parameter)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Idempotency key

**Plain English:** A unique token sent with a write so that retrying it does not repeat the side effect.

**Formal:** key = hash(task, action, payload); the server stores one result per key and returns it on repeats.

**Lesson:** [Building reliable agents](lessons/19-agents/02-reliable-agents.md)

### Inference

**Plain English:** Running a trained model to make predictions on new inputs.

**Formal:** Computing $f_{\theta^*}(x_{\text{new}})$ with fixed parameters; for LLMs, generating tokens from a prompt.

**Example:** Scoring a new transaction for fraud in 20 milliseconds.

**Related:** [Training](#training), [Quantization](#quantization)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### InfoNCE

**Plain English:** The contrastive loss: each item must pick out its true partner from a batch of distractors.

**Formal:** $\ell_i = -\log \dfrac{\exp(\text{sim}(z_i, z_{i^+})/\tau)}{\sum_{k \ne i} \exp(\text{sim}(z_i, z_k)/\tau)}$, cross-entropy over candidates with temperature $\tau$; chance level is the log of the number of candidates.

**Example:** With 2 images (4 views) and $\tau = 0.5$, the lesson's example gives a loss of 0.871 versus a chance level of ln 3 = 1.099.

**Related:** [Cross-entropy](#cross-entropy), [Temperature](#temperature), [Cosine similarity](#cosine-similarity)

**Lesson:** [Self-supervised vision](lessons/25-computer-vision/04-self-supervised-vision.md)

### Intersection over union

**Plain English:** How much two regions overlap, as a fraction of the area they cover together; 1 is identical, 0 is disjoint.

**Formal:** $\text{IoU}(A, B) = |A \cap B| / |A \cup B|$, for boxes or pixel masks.

**Example:** Boxes (50, 50, 150, 150) and (60, 60, 160, 160) overlap in 8100 pixels out of 11,900, so IoU = 0.68.

**Related:** [Non-maximum suppression](#non-maximum-suppression), [Metric](#metric)

**Lesson:** [Object detection and YOLO](lessons/25-computer-vision/01-object-detection-yolo.md)

### Isolation forest

**Plain English:** An anomaly detector that counts how many random cuts it takes to separate a point from the rest; fewer cuts means more anomalous.

**Formal:** $s(x) = 2^{-E[h(x)]/c(\psi)}$, where $h$ is path length in random trees built on subsamples of size $\psi$ and $c(n) = 2(\ln(n-1) + \gamma) - 2(n-1)/n$.

**Example:** With $\psi = 256$, $c = 10.245$; a point isolated in 4 cuts on average scores 0.763.

**Related:** [Anomaly detection](#anomaly-detection)

**Lesson:** [Anomaly detection](lessons/24-applied-ml/05-anomaly-detection.md)

### KL divergence

**Plain English:** The extra surprise you pay for believing distribution Q when reality follows P. Zero only when they match; not symmetric.

**Formal:** $D_{\text{KL}}(P \Vert Q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(P, Q) - H(P) \ge 0$.

**Example:** For $P = [0.7, 0.2, 0.1]$ and $Q = [0.6, 0.25, 0.15]$, KL is 0.023 nats; reversed it is 0.024.

**Related:** [Cross-entropy](#cross-entropy), [Entropy](#entropy), [Distillation](#distillation)

**Lesson:** [Information theory](lessons/00-foundations/05-information-theory.md)

### KV cache

**Plain English:** Saved attention keys and values of earlier tokens, so generation doesn't recompute the whole prefix for every new token.

**Formal:** Cached $K, V$ tensors per layer; each new token computes only its own query against them, $O(n)$ per step.

**Example:** Long contexts and large batches make the KV cache the main GPU-memory cost of serving.

**Related:** [Attention](#attention), [Context window](#context-window)

**Lesson:** [Self-attention](lessons/14-transformers/01-self-attention.md)

### Label

**Plain English:** The correct answer attached to a training example.

**Formal:** The target value $y_i$ paired with features $x_i$ in supervised data.

**Example:** "Spam" or "not spam" for an email.

**Related:** [Feature](#feature), [Loss](#loss)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Label smoothing

**Plain English:** Training against slightly softened targets (a little probability on every wrong class) so the model can't push its logits to infinity.

**Formal:** target $= (1 - \varepsilon)\,\text{one-hot} + \varepsilon / K$.

**Example:** With $\varepsilon = 0.1$ and 10 classes, the true class's target is 0.91 and every other class gets 0.01.

**Related:** [Cross-entropy](#cross-entropy), [Calibration](#calibration), [Regularization](#regularization)

**Lesson:** [Loss functions](lessons/10-neural-networks/02-loss-functions.md)

### Latent diffusion

**Plain English:** Running a diffusion model on a compressed representation from a frozen autoencoder instead of on raw pixels, then decoding once at the end.

**Formal:** Train $E, \mathcal{D}$; set $z_0 = s_f E(x)$; train $\epsilon_\theta(z_t, t, c)$ on $z_t = \sqrt{\bar\alpha_t} z_0 + \sqrt{1 - \bar\alpha_t}\epsilon$; generate $x = \mathcal{D}(z_0 / s_f)$.

**Example:** Stable Diffusion 1.5 maps 512×512×3 images to 64×64×4 latents, 48 times fewer values.

**Related:** [Classifier-free guidance](#classifier-free-guidance), [Flow matching](#flow-matching)

**Lesson:** [Latent diffusion and Stable Diffusion](lessons/13-deep-architectures/05-latent-diffusion-stable-diffusion.md)

### Learning rate

**Plain English:** How big a step the optimizer takes each update.

**Formal:** The scalar $\eta$ in $\theta \leftarrow \theta - \eta\nabla L$; often scheduled (warmup, decay).

**Example:** Too high diverges (loss explodes); too low trains very slowly.

**Related:** [Gradient descent](#gradient-descent), [Hyperparameter](#hyperparameter)

**Lesson:** [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md)

### Learning-rate warmup

**Plain English:** Starting training with a small learning rate and raising it linearly to its peak over the first steps, before decaying it.

**Formal:** $\eta_t = \eta_{\max}\, t / T_w$ for $t < T_w$, usually followed by cosine decay.

**Example:** max 3e-4 with 100 warmup steps gives 1.5e-4 at step 49.

**Lesson:** [Training and regularization](lessons/12-training-regularization/01-training-and-regularization.md)

### Load balancing (MoE)

**Plain English:** Keeping a mixture-of-experts router from sending most tokens to the same few experts.

**Formal:** Either an auxiliary loss $\alpha E \sum_e F_e P_e$ (1 when uniform), or a per-expert selection bias nudged down for overloaded experts and up for underloaded ones, which leaves the gates unchanged.

**Example:** With 8 experts and top-2 on 12 tokens, balancing cut the busiest expert from 8 requests to 4 and dropped assignments from 6 to 2.

**Related:** [Mixture of experts](#mixture-of-experts)

**Lesson:** [Mixture of experts](lessons/15-llms/05-mixture-of-experts.md)

### Local outlier factor (LOF)

**Plain English:** An anomaly score comparing a point's local density with the density around its neighbors.

**Formal:** The ratio of the neighbors' average local reachability density to the point's own; values well above 1 mean the point is in a sparser region than its neighbors.

**Example:** A point just outside a tight cluster gets a high LOF even if it is closer to the cluster than normal points in a wide cluster are to theirs.

**Related:** [Isolation forest](#isolation-forest), [Anomaly detection](#anomaly-detection)

**Lesson:** [Anomaly detection](lessons/24-applied-ml/05-anomaly-detection.md)

### Log-sum-exp trick

**Plain English:** Turning log scores into probabilities without overflow or underflow by subtracting the largest score before exponentiating.

**Formal:** $\log\sum_i e^{s_i} = m + \log\sum_i e^{s_i - m}$ with $m = \max_i s_i$.

**Example:** scores $(-1000, -1001)$ underflow to $0/0$ directly; shifted to $(0, -1)$ they give probabilities $(0.731, 0.269)$.

**Lesson:** [Naive Bayes](lessons/05-instance-and-probabilistic/02-naive-bayes.md)

### Logit

**Plain English:** The raw score a model outputs before it is turned into a probability.

**Formal:** For logistic regression, $z = w\cdot x + b$; the probability is $\sigma(z)$. For multiclass models and LLMs, a vector of logits feeds softmax.

**Example:** A logit of 0 corresponds to probability 0.5.

**Related:** [Softmax](#softmax), [Calibration](#calibration)

**Lesson:** [Logistic regression](lessons/04-classification/01-logistic-regression.md)

### LoRA

**Plain English:** A cheap way to fine-tune a large model by training small add-on matrices instead of all the weights.

**Formal:** $W = W_0 + \frac{\alpha}{r}BA$ with frozen $W_0$ and trainable low-rank $B \in \mathbb{R}^{d_{\text{out}}\times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$.

**Example:** Rank 8 on a 4096 × 4096 matrix trains 65,536 parameters instead of 16.8 million.

**Related:** [Fine-tuning](#fine-tuning)

**Lesson:** [Adapting LLMs](lessons/15-llms/03-adapting-llms.md)

### Loss

**Plain English:** A single number measuring how wrong the model's predictions are, which training tries to minimize.

**Formal:** $L(\theta) = \frac{1}{n}\sum_i \ell(f_\theta(x_i), y_i)$, plus any regularization terms.

**Example:** Mean squared error for regression; cross-entropy for classification.

**Related:** [Metric](#metric), [Gradient descent](#gradient-descent)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Loss scaling

**Plain English:** Multiplying the loss by a big number before backward so tiny fp16 gradients do not round to zero, then dividing it back out.

**Formal:** $\nabla(S\mathcal{L}) = S\nabla\mathcal{L}$. A dynamic scaler skips the step and halves $S$ when any gradient overflows, and doubles $S$ after a run of clean steps (2,000 by default in PyTorch). Not needed with bf16.

**Example:** A gradient of $10^{-8}$ is 0 in fp16; times $2^{16}$ it is $6.55 \times 10^{-4}$, which fp16 can hold.

**Related:** [Mixed precision](#mixed-precision)

**Lesson:** [Gradient clipping and mixed precision](lessons/27-training-at-scale/01-gradient-clipping-amp.md)

### Lost in the middle

**Plain English:** The tendency of language models to use information at the start or end of a long context more reliably than information in the middle.

**Formal:** Answer accuracy as a function of the relevant passage's position in the input is often U-shaped (Liu et al., 2023); the size of the effect varies by model and task.

**Example:** Ordering retrieved chunks so the most relevant sit first and last, with the weakest in the middle, is a common mitigation.

**Related:** [Context engineering](#context-engineering), [Context window](#context-window)

**Lesson:** [Context engineering](lessons/28-prompting-and-context/03-context-engineering.md)

### Markov decision process

**Plain English:** The standard way to write down a sequential decision problem: what the agent sees, what it can do, what happens next, and what score it gets.

**Formal:** A tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ of states, actions, transition probabilities $P(s' \mid s, a)$, rewards, and a discount, where the next state depends only on the current state and action (the Markov property).

**Example:** In a grid world the state is the cell, actions are the four moves, and each move costs −1 until the goal.

**Related:** [Bellman equation](#bellman-equation), [Discount factor](#discount-factor)

**Lesson:** [MDPs, states, actions, and rewards](lessons/26-reinforcement-learning/01-mdps-states-actions-rewards.md)

### MASE

**Plain English:** Mean absolute scaled error: forecast error divided by the error of the seasonal-naive forecast, so it compares across series of any scale.

**Formal:** $\text{MASE} = \text{MAE} / \big(\frac{1}{T-m}\sum_{t>m} |y_t - y_{t-m}|\big)$; below 1 beats seasonal naive.

**Example:** MAE 3.6 against a seasonal-naive MAE of 4.5 gives MASE 0.8.

**Related:** [Baseline](#baseline), [Metric](#metric)

**Lesson:** [Time series](lessons/24-applied-ml/04-time-series.md)

### Masked language modeling

**Plain English:** A pretraining task where some tokens are hidden and the model guesses them using the words on both sides.

**Formal:** Select about 15% of positions $\mathcal{M}$; replace 80% with `[MASK]`, 10% with a random token, keep 10%; minimize $-\frac{1}{\lvert\mathcal{M}\rvert}\sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x})$ with a bidirectional encoder.

**Example:** In "the cat [MASK] on the mat", the words on both sides make "sat" the likely answer.

**Related:** [Attention](#attention), [Cross-entropy](#cross-entropy), [Embedding](#embedding)

**Lesson:** [BERT and masked language modeling](lessons/14-transformers/03-bert-masked-language-modeling.md)

### Matthews correlation coefficient

**Plain English:** A score for a yes/no classifier that uses all four cells of the confusion matrix and is 0 for any constant or random predictor, so it can't be gamed by predicting the majority class.

**Formal:** $\text{MCC} = \dfrac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$, ranging from $-1$ to $+1$.

**Example:** $TP = 27$, $FP = 128$, $FN = 7$, $TN = 638$ gives MCC $= 16330 / 51026.9 = 0.320$, while accuracy is 0.831.

**Related:** [Average precision](#average-precision), [Metric](#metric)

**Lesson:** [Imbalanced data](lessons/24-applied-ml/02-imbalanced-data.md)

### Mean average precision

**Plain English:** The standard detection score: how well confidence ranks correct boxes above wrong ones, averaged over classes (and, for COCO, over IoU thresholds).

**Formal:** AP is the area under the interpolated precision-recall curve at one IoU threshold; mAP averages AP over classes; mAP@0.5:0.95 also averages over IoU thresholds 0.50, 0.55, ..., 0.95.

**Example:** Detections ranked TP, TP, FP, FP, TP against 3 objects give AP@0.5 = 0.867.

**Related:** [Intersection over union](#intersection-over-union), [Metric](#metric)

**Lesson:** [Object detection and YOLO](lessons/25-computer-vision/01-object-detection-yolo.md)

### Mel spectrogram

**Plain English:** A spectrogram whose frequency axis is pooled into bands spaced the way human pitch perception is: narrow at low frequencies, wide at high ones.

**Formal:** $S = \log(\max(P B^\top, \epsilon))$ where $P$ is the power spectrogram and $B$ is a matrix of triangular filters equally spaced on the mel scale $m = 2595 \log_{10}(1 + f/700)$.

**Example:** 80 mel filters turn a $98 \times 257$ power spectrogram into a $98 \times 80$ log-mel spectrogram, the input format of models like Whisper.

**Related:** [Spectrogram](#spectrogram), [MFCC](#mfcc)

**Lesson:** [Spectrograms and mel features](lessons/29-speech-and-audio/01-spectrograms-mel-features.md)

### Metric

**Plain English:** How people judge a model's performance, which can differ from the loss used for training.

**Formal:** Any evaluation function of predictions and labels, possibly non-differentiable and threshold-dependent.

**Example:** RMSE in dollars, recall at a review budget, p95 latency, cost per request.

**Related:** [Loss](#loss), [Baseline](#baseline)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### MFCC

**Plain English:** Mel-frequency cepstral coefficients: a compact summary of each audio frame's spectral shape, made by compressing the log-mel spectrum.

**Formal:** Apply a type-II discrete cosine transform to each log-mel frame and keep the first (usually 13) coefficients; coefficient 0 tracks overall energy.

**Example:** A $98 \times 80$ log-mel spectrogram becomes a $98 \times 13$ MFCC matrix.

**Related:** [Mel spectrogram](#mel-spectrogram)

**Lesson:** [Spectrograms and mel features](lessons/29-speech-and-audio/01-spectrograms-mel-features.md)

### Minimum detectable effect

**Plain English:** The smallest true improvement an experiment is designed to detect reliably.

**Formal:** The effect size $\delta$ at which the test has the chosen power $1 - \beta$ at significance $\alpha$; the sample size per arm scales as $1/\delta^2$.

**Example:** Detecting 10% → 11% conversion (a 10% relative MDE) at $\alpha = 0.05$ and power 0.8 takes 14,749 users per arm; halving the MDE takes about four times as many.

**Related:** [Statistical power](#statistical-power), [P-value](#p-value)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Mixed precision

**Plain English:** Training with most math in a 16-bit format (bf16 or fp16) for speed and memory, while keeping master weights and sensitive reductions in float32.

**Formal:** bf16 has float32's 8 exponent bits and 7 fraction bits; fp16 has 5 and 10 (max 65,504), so fp16 needs dynamic loss scaling to keep small gradients from flushing to zero.

**Example:** `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` runs matmuls in bf16 without a loss scaler.

**Related:** [Quantization](#quantization), [Log-sum-exp trick](#log-sum-exp-trick)

**Lesson:** [Numerical stability](lessons/00-foundations/06-numerical-stability.md)

### Mixture of experts

**Plain English:** A model whose feed-forward layers are split into many "experts", with a router sending each token to only a few of them, so it has many parameters but uses few per token.

**Formal:** Each MoE layer has $E$ expert networks and a router that picks the top $k$ per token. Total parameters set memory; active parameters (shared plus $k$ experts) set compute per token.

**Example:** 4B shared + 16 experts × 2B with top-2 routing: 36B total (72 GB in BF16), 8B active per token.

**Related:** [Parameter](#parameter), [Inference](#inference)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Mode collapse

**Plain English:** A generator failure where it produces only a few kinds of output and ignores parts of the data distribution.

**Formal:** $p_g$ concentrates on a subset of the modes of $p_{\text{data}}$; shows up as low recall or coverage even when sample quality and losses look fine.

**Example:** A GAN trained on two clusters at -2 and +2 puts every sample near +2.

**Related:** [GAN](#gan)

**Lesson:** [GANs, generator versus discriminator](lessons/13-deep-architectures/04-gans-generator-discriminator.md)

### Model Context Protocol (MCP)

**Plain English:** An open standard for connecting AI applications to tools and data, so one integration works with many apps.

**Formal:** MCP clients inside a host application talk JSON-RPC (over stdio or HTTP) to MCP servers, which expose tools (callable functions), resources (readable data), and prompts (templates).

**Example:** An issue-tracker MCP server lets any compliant IDE or agent search and create tickets without custom glue code.

**Related:** [Tool calling](#tool-calling), [Agent](#agent), [Prompt injection](#prompt-injection)

**Lesson:** [Tool use and agents](lessons/19-agents/01-tool-use-and-agents.md)

### Multi-agent system

**Plain English:** Several LLM agents, each with its own prompt and context, that split a task and pass messages, usually coordinated by an orchestrator.

**Formal:** Designs range from fixed pipelines and fan-out/fan-in to orchestrator-worker (a model plans subtasks and delegates them) and peer swarms. Parallel workers cut the critical path from $ks$ to about $s + 2$ calls; total tokens depend on how much each worker re-explores.

**Example:** An orchestrator sends four research briefs to parallel workers and synthesizes their 400-token summaries.

**Related:** [Agent](#agent), [ReAct](#react)

**Lesson:** [Multi-agent systems](lessons/19-agents/05-multi-agent-systems.md)

### Multi-head latent attention

**Plain English:** DeepSeek's attention variant that caches one small compressed vector per token instead of full keys and values for every head.

**Formal:** Hidden states are projected down to a latent of size $r$ (`kv_lora_rank`), cached together with a shared RoPE key, and projected back up to per-head keys and values at attention time.

**Example:** DeepSeek-V3 caches $(512 + 64) \cdot 2 \cdot 61 = 70{,}272$ bytes per token, 71× less than full keys and values for 128 heads.

**Related:** [Grouped-query attention](#grouped-query-attention), [KV cache](#kv-cache)

**Lesson:** [Open models: architecture walkthroughs](lessons/15-llms/07-open-models-architecture-walkthroughs.md)

### Mutual information

**Plain English:** How much knowing one variable reduces uncertainty about another; catches non-linear relationships that correlation misses.

**Formal:** $I(X;Y) = \sum_{x,y} p(x,y) \ln \frac{p(x,y)}{p(x)p(y)}$, which is 0 exactly when $X$ and $Y$ are independent.

**Example:** For $y = x^2$ with $x$ symmetric around 0, correlation is about 0 but mutual information is large.

**Related:** [Feature](#feature)

**Lesson:** [Feature engineering](lessons/24-applied-ml/01-feature-engineering.md)

### Named entity recognition

**Plain English:** Finding names of people, organizations, places, and other typed things in text, as spans.

**Formal:** A sequence-labeling task mapping tokens $x_1..x_n$ to tags $y_1..y_n$ (usually BIO), evaluated with entity-level precision, recall, and F1 on exact span and type matches.

**Example:** "Ada Lovelace joined Acme Corp" yields (Ada Lovelace, PER) and (Acme Corp, ORG).

**Related:** [BIO tagging](#bio-tagging), [Viterbi algorithm](#viterbi-algorithm)

**Lesson:** [Named entity recognition](lessons/09-classical-nlp/03-named-entity-recognition.md)

### Nested cross-validation

**Plain English:** Cross-validation wrapped around a whole tuning procedure, so the reported score isn't inflated by picking the best of many trials.

**Formal:** Outer folds hold out test data; inside each, an inner CV selects hyperparameters; the outer scores are averaged.

**Example:** A search's best inner score of 0.88 next to a nested score of 0.85 shows the selection optimism.

**Related:** [Hyperparameter](#hyperparameter), [Overfitting](#overfitting)

**Lesson:** [Hyperparameter tuning](lessons/24-applied-ml/03-hyperparameter-tuning.md)

### Non-maximum suppression

**Plain English:** The cleanup step after a detector: keep the highest-scoring box, delete boxes that overlap it too much, repeat.

**Formal:** Greedy: sort boxes by score; repeatedly keep the top box and remove every remaining box with IoU above a threshold (often 0.45 to 0.7), usually per class.

**Example:** Three boxes on one object with IoUs above 0.6 to the best one collapse to that single box at threshold 0.5.

**Related:** [Intersection over union](#intersection-over-union)

**Lesson:** [Object detection and YOLO](lessons/25-computer-vision/01-object-detection-yolo.md)

### Normal equation

**Plain English:** The closed-form solution to least-squares linear regression, found by setting the gradient of the squared error to zero.

**Formal:** $X^\top X\,\theta = X^\top y$, solved with a linear solver rather than an explicit inverse.

**Example:** For $x = [1, 2, 3]$, $y = [2, 4, 5]$ it gives $w = 1.5$, $b = 0.667$.

**Lesson:** [Linear regression](lessons/03-regression/01-linear-regression.md)

### One-hot encoding

**Plain English:** Turning a category into one 0/1 column per possible value, so no fake ordering is implied.

**Formal:** A category $c$ among $K$ values maps to the unit vector $e_c \in \{0,1\}^K$.

**Example:** color = green among {red, green, blue} becomes $[0, 1, 0]$.

**Related:** [Target encoding](#target-encoding), [Feature](#feature)

**Lesson:** [Feature engineering](lessons/24-applied-ml/01-feature-engineering.md)

### Out-of-bag error

**Plain English:** A free validation estimate for a random forest: score each row using only the trees that never saw it during training.

**Formal:** For row $i$, average the predictions of trees whose bootstrap sample excluded $i$; compute the metric over all rows.

**Example:** In scikit-learn, `RandomForestClassifier(oob_score=True)` exposes it as `oob_score_`.

**Related:** [Bootstrap sample](#bootstrap-sample)

**Lesson:** [Random forests and gradient boosting](lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md)

### Overfitting

**Plain English:** When a model learns the training data's noise and does worse on new data.

**Formal:** A large gap between training and validation performance, typically from high variance.

**Example:** A degree-15 polynomial with training MSE 0.056 and validation MSE 0.52.

**Related:** [Regularization](#regularization), [Bias-variance trade-off](#bias-variance-trade-off)

**Lesson:** [Overfitting and bias-variance](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md)

### P-value

**Plain English:** How surprising the observed result would be if the change actually did nothing.

**Formal:** $P(\text{test statistic at least as extreme as observed} \mid H_0)$. It is not $P(H_0 \mid \text{data})$.

**Example:** $z = 2.825$ in a two-sided test gives $p = 2(1 - \Phi(2.825)) = 0.0047$.

**Related:** [Statistical power](#statistical-power), [Sample ratio mismatch](#sample-ratio-mismatch)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Paged attention

**Plain English:** Storing each sequence's KV cache in small fixed-size blocks allocated as needed, like virtual-memory pages, instead of one big reserved region.

**Formal:** A per-sequence block table maps logical token positions to physical KV blocks (for example 16 tokens each). Waste is under one block per sequence, and blocks can be shared between sequences with a common prefix.

**Example:** With a 40 GB KV budget and 600-token requests, reserving 4,096 tokens each fits 74 sequences; 16-token blocks fit 501.

**Related:** [KV cache](#kv-cache), [Continuous batching](#continuous-batching), [Prefix caching](#prefix-caching)

**Lesson:** [Serving LLMs](lessons/18-llm-inference/02-serving-llms.md)

### Parameter

**Plain English:** A number the model learns from data during training.

**Formal:** An entry of $\theta$, the vector of values the optimizer adjusts to minimize the loss $L(\theta)$.

**Example:** The slope $w$ and bias $b$ in $\hat{y} = wx + b$.

**Related:** [Hyperparameter](#hyperparameter), [Gradient](#gradient)

**Lesson:** [Linear regression](lessons/03-regression/01-linear-regression.md)

### pass@k

**Plain English:** The chance that at least one of $k$ generated programs passes the tests.

**Formal:** From $n$ samples with $c$ passing, the unbiased estimate is $1 - \binom{n-c}{k} / \binom{n}{k}$.

**Example:** $n = 20$, $c = 3$: pass@1 = 0.15, pass@5 = 0.601.

**Related:** [Calibration](#calibration)

**Lesson:** [Evaluating LLM systems](lessons/17-llm-evaluation/01-llm-evaluation.md)

### pass^k

**Plain English:** The chance an agent gets a task right on every one of $k$ independent tries, a measure of consistency.

**Formal:** For per-trial success rate $s$, $\text{pass}^k = s^k$, versus $\text{pass@}k = 1 - (1-s)^k$ (at least one success).

**Example:** With $s = 0.7$, pass^3 $= 0.343$ while pass@3 $= 0.973$.

**Related:** [Agent](#agent), [Metric](#metric)

**Lesson:** [Building reliable agents](lessons/19-agents/02-reliable-agents.md)

### Per-channel quantization

**Plain English:** Each row (output channel) of a weight matrix gets its own scale, so one large weight doesn't crush the precision of the rest.

**Formal:** s_i = max_j |W_ij| / 127, q_ij = round(W_ij / s_i).

**Related:** [Quantization](#quantization)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Perceptron

**Plain English:** A single neuron with a hard threshold that outputs 1 if its weighted sum is at least 0, trained by nudging weights toward misclassified points.

**Formal:** $\hat{y} = [w \cdot x + b \ge 0]$; on a mistake, $w \leftarrow w + \eta (y - \hat{y}) x$, $b \leftarrow b + \eta (y - \hat{y})$.

**Example:** It learns AND in a few passes but never converges on XOR, because no line separates XOR's classes.

**Lesson:** [The neural-network forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md)

### Permutation test

**Plain English:** A significance test that shuffles group labels to see how often chance alone produces a difference as large as the one observed.

**Formal:** p = (number of shuffles with |difference| ≥ observed + 1) / (number of shuffles + 1).

**Example:** Comparing per-example scores of two models: p = 0.004 means about 4 in 1,000 random relabelings beat the real gap.

**Lesson:** [Probability and statistics](lessons/00-foundations/03-probability-and-statistics.md)

### Perplexity

**Plain English:** How "surprised" a language model is by text, roughly the number of tokens it's choosing between.

**Formal:** $\exp$ of the mean next-token cross-entropy.

**Example:** Mean loss 1.25 gives perplexity 3.49; uniform guessing over 50,000 tokens gives 50,000.

**Related:** [Cross-entropy](#cross-entropy), [Token](#token)

**Lesson:** [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md)

### Pipeline parallelism

**Plain English:** Giving each GPU a block of consecutive layers, like stations on an assembly line, and streaming micro-batches through.

**Formal:** With $p$ stages and $m$ micro-batches, a GPipe schedule is idle for $(p-1)/(m+p-1)$ of the step (the pipeline bubble). Only boundary activations cross between GPUs.

**Example:** 4 stages and 16 micro-batches leave $3/19 \approx 16\%$ of the schedule idle.

**Related:** [Tensor parallelism](#tensor-parallelism), [Data parallelism](#data-parallelism)

**Lesson:** [Scaling and distributed training](lessons/27-training-at-scale/04-scaling-distributed.md)

### Point-in-time join

**Plain English:** Joining each training label to the feature values as they were at the label's timestamp, never later ones.

**Formal:** For label (entity e, time t), use the feature row for e with the largest timestamp s ≤ t (optionally s ≥ t − TTL).

**Example:** A churn label at day 70 uses the day-70 snapshot of "days since last purchase", not the day-175 one.

**Related:** [Training-serving skew](#training-serving-skew), [Feature store](#feature-store), [Data leakage](#data-leakage)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

### Policy gradient

**Plain English:** A way to train a policy directly: make actions that led to better-than-expected outcomes more likely, and worse ones less likely.

**Formal:** $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[\sum_t A_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\big]$, derived with the log-derivative trick $\nabla P = P \nabla \log P$. REINFORCE uses $A_t = G_t - b(s_t)$.

**Example:** With logits $[0, 0.5, -0.5, 1]$ and action 4 taken, $\nabla_z \log \pi = [-0.167, -0.276, -0.102, 0.545]$; scaling by an advantage of $+0.79$ raises action 4's logit.

**Related:** [Advantage](#advantage), [PPO](#ppo), [Softmax](#softmax)

**Lesson:** [Policy gradients and REINFORCE](lessons/26-reinforcement-learning/04-policy-gradients-reinforce.md)

### Population Stability Index

**Plain English:** A score for how much a feature's distribution has moved between a reference window and a current window.

**Formal:** $\text{PSI} = \sum_b (a_b - e_b)\ln(a_b / e_b)$ over bins (usually reference deciles), with $e_b$ and $a_b$ the reference and current shares. Conventional bands: under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 significant.

**Example:** Two decile bins falling to 5% and one rising to 20% gives PSI 0.139.

**Related:** [Drift](#drift), [Concept drift](#concept-drift)

**Lesson:** [Model monitoring and drift](lessons/22-ml-system-design/03-model-monitoring-and-drift.md)

### Position bias

**Plain English:** Users interact with items partly because of where they are shown, so top slots get more clicks regardless of quality.

**Formal:** $P(\text{click} \mid \text{item}, \text{position}) \approx P(\text{examined} \mid \text{position}) \cdot P(\text{relevant} \mid \text{item})$; training on raw clicks confounds the two factors.

**Example:** A ranker trained on raw clicks learns to promote whatever the previous model put first, creating a feedback loop; adding position as a feature fixed at serving time, or inverse propensity weighting, corrects for it.

**Related:** [Cold start](#cold-start), [Calibration](#calibration)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

### PPO

**Plain English:** Proximal policy optimization: a policy-gradient method that reuses each batch for several updates but stops rewarding any action's probability change beyond about ±20%.

**Formal:** Maximizes $\mathbb{E}_t[\min(r_t \hat A_t, \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat A_t)]$ with $r_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$, plus a value loss and often an entropy bonus.

**Example:** With $\varepsilon = 0.2$, a sample with $r = 1.5$ and $\hat A = 2$ scores $\min(3.0, 2.4) = 2.4$ and contributes no gradient.

**Related:** [Policy gradient](#policy-gradient), [Generalized advantage estimation](#generalized-advantage-estimation), [GRPO](#grpo), [DPO](#dpo)

**Lesson:** [PPO](lessons/26-reinforcement-learning/05-ppo.md)

### Prefill

**Plain English:** The first phase of generation, where the model processes the whole prompt in parallel, fills the KV cache, and produces the first output token.

**Formal:** One forward pass over $N$ prompt tokens, about $2PN$ FLOPs for a $P$-parameter model. Its arithmetic intensity is about $N$ FLOPs per byte, so it is compute-bound, unlike decode.

**Example:** A 2,000-token prompt on an 8B model needs about $3.2 \times 10^{13}$ FLOPs, 32 ms at a peak of $10^{15}$ FLOP/s.

**Related:** [KV cache](#kv-cache), [Time to first token](#time-to-first-token), [Inference](#inference)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Prefix caching

**Plain English:** Reusing the saved KV cache for a prompt beginning that many requests share, so it isn't recomputed each time.

**Formal:** KV blocks are keyed by the exact token prefix they encode; a new request whose prompt starts with a cached prefix skips prefill for those tokens and shares the blocks.

**Example:** A 2,000-token shared system prompt plus 200 user tokens: a cache hit prefills 200 tokens instead of 2,200 (90.9% less).

**Related:** [Prefill](#prefill), [Paged attention](#paged-attention), [KV cache](#kv-cache)

**Lesson:** [Serving LLMs](lessons/18-llm-inference/02-serving-llms.md)

### Product quantization

**Plain English:** Compressing vectors by cutting each into pieces and storing, for every piece, only the ID of its nearest prototype.

**Formal:** Split $x \in \mathbb{R}^d$ into $m$ subvectors, learn 256 centroids per position with k-means, store $m$ one-byte codes; query distances are sums of $m$ precomputed table lookups.

**Example:** A 768-dimensional float32 vector (3,072 bytes) with $m = 96$ takes 96 bytes, 32 times smaller.

**Related:** [BM25](#bm25)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Prompt injection

**Plain English:** Text that tries to override a model's instructions, typed by a user or hidden in content the model reads.

**Formal:** Adversarial input exploiting the model's inability to separate trusted instructions from untrusted data; indirect injection arrives via retrieved documents, pages, emails, or tool outputs.

**Example:** A web page containing hidden text telling the assistant to email the user's invoices elsewhere.

**Related:** [Agent](#agent), [Tool calling](#tool-calling)

**Lesson:** [Securing AI systems](lessons/21-safety-security/01-ai-security.md)

### Q-learning

**Plain English:** An off-policy TD method that learns the value of acting greedily, whatever exploring the agent does while it learns.

**Formal:** $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$; converges to $Q^*$ if every pair keeps being visited.

**Example:** $Q = -2$, $r = -1$, best next value $-1.5$, $\gamma = 0.9$, $\alpha = 0.5$: the new value is $-2.175$.

**Related:** [SARSA](#sarsa), [Epsilon-greedy](#epsilon-greedy), [Temporal-difference learning](#temporal-difference-learning)

**Lesson:** [Q-learning and SARSA](lessons/26-reinforcement-learning/02-q-learning-sarsa.md)

### Q-value

**Plain English:** How much total reward you expect if you take a particular action in a particular state and then keep following your policy.

**Formal:** $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$; the state value is $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$.

**Example:** Acting greedily means picking $\arg\max_a Q(s, a)$.

**Related:** [Q-learning](#q-learning), [Bellman equation](#bellman-equation)

**Lesson:** [MDPs, states, actions, and rewards](lessons/26-reinforcement-learning/01-mdps-states-actions-rewards.md)

### QLoRA

**Plain English:** LoRA fine-tuning on top of a base model stored in 4-bit precision, so large models fit on a single smaller GPU.

**Formal:** Frozen base weights stored in 4-bit NF4 with blockwise (double-quantized) scales, dequantized to bf16 on the fly; only 16-bit LoRA matrices $A, B$ receive gradients.

**Example:** A synthetic 7B-style model needs about 108.6 GB to fully fine-tune but about 5.3 GB with QLoRA at rank 16.

**Related:** [LoRA](#lora), [Quantization](#quantization)

**Lesson:** [Fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md)

### Quantization

**Plain English:** Storing model weights with fewer bits to save memory and often speed up inference.

**Formal:** Mapping weights from 16 or 32-bit floats to 8 or 4-bit integers with scale factors; memory = parameters × bits / 8.

**Example:** A 7B model needs 14 GB in fp16 and 3.5 GB in int4.

**Related:** [Distillation](#distillation), [Inference](#inference)

**Lesson:** [Reliability, cost, and observability](lessons/20-production-ai/02-reliability-cost-and-observability.md)

### Query rewriting

**Plain English:** Changing the user's question before retrieval so it uses the words and shape the documents use.

**Formal:** Common forms: multi-query (several paraphrases, results fused), HyDE (embed a model-written hypothetical answer), and decomposition (split into sub-questions).

**Example:** "Can I get reimbursed for a game I never opened?" gains "refund", "digital downloads", and "accessed", and now matches the refund policy.

**Related:** [BM25](#bm25)

**Lesson:** [RAG pipeline](lessons/16-rag/01-rag-pipeline.md)

### RAG

**Plain English:** Retrieval-augmented generation: look up relevant documents at query time and give them to the model to answer from.

**Formal:** A pipeline of ingestion, chunking, indexing, retrieval, optional reranking, context construction, and grounded generation; it changes the model's context, not its weights.

**Example:** A support bot retrieves the current refund policy and cites it.

**Related:** [Grounding](#grounding), [Fine-tuning](#fine-tuning)

**Lesson:** [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md)

### Random search

**Plain English:** Trying hyperparameter settings sampled at random from chosen ranges instead of a fixed grid.

**Formal:** Draw $n$ configurations independently; if a fraction $p$ of the space is good, $P(\text{at least one good}) = 1 - (1-p)^n$.

**Example:** With $p = 0.05$, 60 trials give $1 - 0.95^{60} \approx 0.95$.

**Related:** [Hyperparameter](#hyperparameter), [Bayesian optimization](#bayesian-optimization)

**Lesson:** [Hyperparameter tuning](lessons/24-applied-ml/03-hyperparameter-tuning.md)

### ReAct

**Plain English:** An agent pattern where the model alternates short reasoning notes with tool calls and reads each result before deciding the next step.

**Formal:** The trajectory interleaves thought, action, and observation: $(\tau_1, a_1, o_1, \tau_2, a_2, o_2, \dots)$, each generated conditioned on everything before it.

**Example:** "UA412 is cheapest; check refundability" → `get_fare_rules(UA412)` → `{"refundable": false}` → "check AS330".

**Related:** [Agent](#agent), [Tool calling](#tool-calling)

**Lesson:** [Tool use and agents](lessons/19-agents/01-tool-use-and-agents.md)

### Receptive field

**Plain English:** The patch of the input image that can influence one output pixel of a CNN layer.

**Formal:** Each layer adds $(k - 1) \times$ (product of earlier strides) pixels.

**Example:** Three stacked 3×3 stride-1 convolutions have a 7×7 receptive field.

**Lesson:** [Convolutional networks](lessons/13-deep-architectures/01-convolutional-networks.md)

### Reciprocal rank fusion

**Plain English:** A way to merge ranked lists from different search methods using only each document's rank, not its score.

**Formal:** $\text{RRF}(d) = \sum_j \frac{1}{c + \text{rank}_j(d)}$, commonly $c = 60$.

**Example:** A document ranked 1st by dense search and 3rd by BM25 scores $1/61 + 1/63 = 0.0323$.

**Related:** [BM25](#bm25), [RAG](#rag)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Reduce-scatter

**Plain English:** Summing tensors across GPUs but giving each GPU only its own slice of the result.

**Formal:** Each of $N$ ranks ends with the reduced values of one $1/N$ chunk; it sends $\frac{N-1}{N}T$ bytes per rank. The first half of a ring all-reduce.

**Example:** In ZeRO, each GPU receives the summed gradient for only the parameters whose optimizer state it owns.

**Related:** [All-reduce](#all-reduce), [All-gather](#all-gather), [ZeRO](#zero)

**Lesson:** [Collective operations from scratch](lessons/27-training-at-scale/03-collective-ops-from-scratch.md)

### Regularization

**Plain English:** Anything that discourages a model from memorizing, so it generalizes better.

**Formal:** Penalties ($\lambda\lVert w\rVert^2$, $\lambda\lVert w\rVert_1$) or procedures (dropout, early stopping, augmentation) that reduce variance.

**Example:** Ridge regression shrinks the slope from 1.5 to 0.6 with $\lambda = 1$ in the course example.

**Related:** [Overfitting](#overfitting)

**Lesson:** [Regularization and regression metrics](lessons/03-regression/02-regularization-and-regression-metrics.md)

### Reparameterization trick

**Plain English:** Writing a random sample as a deterministic function of the model's outputs plus independent noise, so gradients can flow through the sampling step.

**Formal:** $z = \mu + \sigma \epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$.

**Example:** $\mu = 1$, $\sigma = 1$, $\epsilon = 0.5$ gives $z = 1.5$.

**Lesson:** [Autoencoders, diffusion, and transfer learning](lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md)

### Reward model

**Plain English:** A model that reads a response and outputs one number saying how much people would prefer it.

**Formal:** Trained on preference pairs with the Bradley-Terry loss $-\log \sigma(r(y_w) - r(y_l))$, where $y_w$ is the chosen and $y_l$ the rejected response.

**Example:** In RLHF, PPO updates the policy to raise the reward model's score while a KL penalty keeps it near the reference model.

**Related:** [Calibration](#calibration)

**Lesson:** [Adapting LLMs](lessons/15-llms/03-adapting-llms.md)

### RMSNorm

**Plain English:** A simpler layer normalization that rescales activations by their root-mean-square, without subtracting the mean or adding a bias.

**Formal:** $\text{RMSNorm}(x) = \gamma \odot x / \sqrt{\tfrac{1}{h}\sum_i x_i^2 + \epsilon}$, with one learned scale vector $\gamma$ of size $h$.

**Example:** Llama 3 8B has 65 RMSNorms of 4,096 parameters each (two per block plus a final one).

**Related:** [SwiGLU](#swiglu), [RoPE](#rope)

**Lesson:** [Open models: architecture walkthroughs](lessons/15-llms/07-open-models-architecture-walkthroughs.md)

### Robust z-score

**Plain English:** A z-score that uses the median and median absolute deviation, so outliers can't hide by inflating the mean and spread.

**Formal:** $z^{\text{robust}} = 0.6745\,(x - \tilde x)/\text{MAD}$, with $\text{MAD} = \text{median}|x_i - \tilde x|$; a common cutoff is 3.5.

**Example:** In $[10, 11, 12, 12, 13, 14, 50]$, 50 has plain z 2.44 but robust z 25.6.

**Related:** [Anomaly detection](#anomaly-detection)

**Lesson:** [Anomaly detection](lessons/24-applied-ml/05-anomaly-detection.md)

### RoPE

**Plain English:** Rotary position embedding: positions are encoded by rotating each query and key by an angle that grows with position, so attention scores depend on relative distance.

**Formal:** Pairs of dimensions $(x_{2i}, x_{2i+1})$ at position $m$ are rotated by angle $m\theta_i$ with $\theta_i = \text{base}^{-2i/d}$ (`rope_theta` is the base). Then $q_m \cdot k_n$ depends only on $m - n$.

**Example:** Llama 3 uses `rope_theta = 500000`; methods such as YaRN rescale the angles to extend context.

**Related:** [Attention](#attention)

**Lesson:** [Open models: architecture walkthroughs](lessons/15-llms/07-open-models-architecture-walkthroughs.md)

### Rotary position embedding (RoPE)

**Plain English:** A way to give attention position information by rotating each pair of query and key features by an angle proportional to the token's position.

**Formal:** Rotating $q$ by $p\theta$ and $k$ by $m\theta$ makes $q \cdot k$ depend only on the offset $p - m$.

**Example:** Used by most current open LLMs in place of a learned position table.

**Lesson:** [The transformer architecture](lessons/14-transformers/02-transformer-architecture.md)

### Sample ratio mismatch

**Plain English:** When the number of users in each arm of an experiment differs from the planned split by more than chance allows, a sign that assignment or logging is broken.

**Formal:** Tested with a chi-square goodness-of-fit test on arm counts against the designed allocation; a tiny p-value (often below 0.001) invalidates the experiment's readout.

**Example:** 50,600 versus 49,400 users in a 50/50 test gives $\chi^2 = 14.4$ and $p \approx 0.00015$.

**Related:** [P-value](#p-value)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### SARSA

**Plain English:** An on-policy TD method that learns the value of the policy it is actually running, exploration included.

**Formal:** $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma Q(s', a') - Q(s, a)]$, where $a'$ is the action the agent will actually take next.

**Example:** On a cliff grid, SARSA learns a path one row away from the edge because its targets include its own random moves.

**Related:** [Q-learning](#q-learning), [Epsilon-greedy](#epsilon-greedy)

**Lesson:** [Q-learning and SARSA](lessons/26-reinforcement-learning/02-q-learning-sarsa.md)

### Scaling law

**Plain English:** A measured power-law relationship that predicts a model's loss from its size, its training data, or its training compute.

**Formal:** The Chinchilla form is $L(N, D) = E + A/N^\alpha + B/D^\beta$, with training compute $C \approx 6ND$.

**Example:** With Hoffmann et al.'s fitted constants, a 70B model on 1.4T tokens is predicted to reach loss 1.937.

**Related:** [Chinchilla-optimal](#chinchilla-optimal), [Cross-entropy](#cross-entropy)

**Lesson:** [Scaling laws](lessons/15-llms/06-scaling-laws.md)

### Self-consistency

**Plain English:** Sampling several reasoning chains and returning the most common final answer.

**Formal:** With $N$ independent chains each correct with probability $p$, a strict majority is correct with probability $\sum_{k > N/2} \binom{N}{k} p^k (1-p)^{N-k}$; plurality voting can do better when wrong answers scatter.

**Example:** $p = 0.7$, $N = 5$ gives at least 0.837; if wrong chains agree and $p = 0.4$, voting drops to 0.317.

**Related:** [Chain-of-thought](#chain-of-thought), [Temperature](#temperature)

**Lesson:** [Few-shot and chain-of-thought](lessons/28-prompting-and-context/02-few-shot-cot.md)

### Self-training

**Plain English:** A semi-supervised recipe: train on the few labeled rows, label the unlabeled rows the model is confident about, add them, and retrain.

**Formal:** Repeat: fit $f$ on $\mathcal{D}_L$; for $x \in \mathcal{D}_U$ with $\max_k P_f(k \mid x) \ge \tau$, add $(x, \arg\max_k P_f(k \mid x))$ to $\mathcal{D}_L$.

**Example:** With 6 labels and threshold 0.9 on two well-separated synthetic blobs, self-training reaches about the same accuracy as using all 200 labels.

**Lesson:** [Learning paradigms](lessons/01-ml-vocabulary/02-learning-paradigms.md)

### Semantic cache

**Plain English:** Reuse a stored answer when a new query is similar enough to a stored one.

**Formal:** Serve the cached answer if cos(embed(q), embed(q_stored)) ≥ τ.

**Lesson:** [Reliability, cost, and observability](lessons/20-production-ai/02-reliability-cost-and-observability.md)

### Sequential testing

**Plain English:** An A/B test design that lets you look at results repeatedly and stop early without inflating false positives.

**Formal:** An always-valid procedure such as the mixture SPRT: stop when the mixture likelihood ratio Λ_n ≥ 1/α; under H0 the chance Λ_n ever crosses is at most α (Ville's inequality).

**Example:** In 4,000 synthetic A/A tests with 14 daily looks, naive peeking gave a 0.216 false-positive rate and mSPRT gave 0.013.

**Related:** [p-value](#p-value), [Statistical power](#statistical-power)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Shadow traffic

**Plain English:** Mirror real requests to a candidate and log its output without returning it to users.

**Lesson:** [Production architecture](lessons/20-production-ai/01-production-architecture.md)

### Silhouette score

**Plain English:** How well a point fits its own cluster compared with the nearest other cluster, averaged over all points. Near 1 means tight, well-separated clusters; near 0 means points sit on boundaries; negative means likely misassigned.

**Formal:** $s_i = (b_i - a_i)/\max(a_i, b_i)$, where $a_i$ is the mean distance from point $i$ to the rest of its cluster and $b_i$ the mean distance to the nearest other cluster.

**Example:** $a_i = 1$, $b_i = 4$ gives $s_i = 3/4 = 0.75$.

**Lesson:** [K-means clustering](lessons/07-unsupervised/01-k-means.md)

### Singular value decomposition (SVD)

**Plain English:** Any matrix factors into a rotation, a stretch along perpendicular axes, and another rotation. On centered data, the axes are the principal components.

**Formal:** $X = U S V^\top$ with orthonormal columns in $U$ and $V$ and nonnegative singular values $s_k$ on the diagonal of $S$. For centered data, PCA variances are $\lambda_k = s_k^2/(n-1)$.

**Example:** five centered points with PCA variance $\lambda_1 = 3.771$ have $s_1^2 = 4 \times 3.771 = 15.08$, so $s_1 \approx 3.88$.

**Lesson:** [PCA, t-SNE, and UMAP](lessons/08-dimensionality-reduction/01-pca.md)

### Sliding-window attention

**Plain English:** Attention where each token looks only at the last few tokens instead of the whole sequence.

**Formal:** Query $i$ may attend key $j$ only if $j \le i$ and $i - j < w$. Cost is $O(nw)$ per layer, the KV cache holds $w$ tokens per layer, and $L$ stacked layers reach $L(w-1)+1$ positions.

**Example:** With $n = 16$ and $w = 4$, the mask has 58 cells instead of 136 for causal attention.

**Related:** [Attention](#attention), [KV cache](#kv-cache), [Context window](#context-window)

**Lesson:** [Attention variants](lessons/14-transformers/04-attention-variants.md)

### SMOTE

**Plain English:** A way to add synthetic examples of a rare class by placing new points between real rare-class points that are close together.

**Formal:** For a minority point $x_i$ and one of its $k$ nearest minority neighbors $x_j$, SMOTE (Synthetic Minority Over-sampling Technique) creates $x_i + \lambda (x_j - x_i)$ with $\lambda \sim U[0, 1)$.

**Example:** From $(1.0, 2.0)$ and neighbor $(1.5, 2.5)$ with $\lambda = 0.4$, SMOTE adds $(1.2, 2.2)$. Apply it only inside training folds.

**Related:** [Data leakage](#data-leakage), [Class weights](#class-weights)

**Lesson:** [Imbalanced data](lessons/24-applied-ml/02-imbalanced-data.md)

### Soft-thresholding

**Plain English:** The lasso's update for one weight: shrink it toward zero by a fixed amount, and set it to exactly zero if it would cross zero.

**Formal:** $S(z, g) = \operatorname{sign}(z)\max(|z| - g, 0)$.

**Example:** $S(1.0, 0.25) = 0.75$, $S(-1.0, 0.25) = -0.75$, $S(0.2, 0.25) = 0$.

**Related:** [Coordinate descent](#coordinate-descent)

**Lesson:** [Regularization and regression metrics](lessons/03-regression/02-regularization-and-regression-metrics.md)

### Softmax

**Plain English:** Turns a list of scores into probabilities that are all positive and add up to 1.

**Formal:** $p_i = e^{z_i/T}/\sum_j e^{z_j/T}$ with temperature $T$.

**Example:** Logits $[2, 1, 0.1]$ become $[0.659, 0.242, 0.099]$.

**Related:** [Logit](#logit), [Temperature](#temperature)

**Lesson:** [Logistic regression](lessons/04-classification/01-logistic-regression.md)

### Spectrogram

**Plain English:** A time-by-frequency picture of sound: how much energy each frequency has in each short slice of time.

**Formal:** The squared magnitude of the short-time Fourier transform: frame the waveform (window $W$, hop $H$), multiply by a window function, FFT each frame; shape $F \times (N_{\text{fft}}/2 + 1)$.

**Example:** One second at 16 kHz with a 400-sample window, 160-sample hop, and 512-point FFT gives a $98 \times 257$ spectrogram.

**Related:** [Mel spectrogram](#mel-spectrogram), [MFCC](#mfcc)

**Lesson:** [Spectrograms and mel features](lessons/29-speech-and-audio/01-spectrograms-mel-features.md)

### Speculative decoding

**Plain English:** A cheap draft model guesses several next tokens, and the large model checks them all in one pass, keeping the ones it agrees with.

**Formal:** With draft length $\gamma$ and per-token acceptance rate $\alpha$, the expected tokens per target pass is $(1 - \alpha^{\gamma+1})/(1 - \alpha)$. With rejection-sampling verification, outputs follow the target model's distribution exactly.

**Example:** $\alpha = 0.8$, $\gamma = 4$: 3.36 tokens per target pass; with draft cost 0.05 per token, a 2.8× speedup.

**Related:** [Inference](#inference), [Temperature](#temperature)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Spotlighting

**Plain English:** Visibly mark untrusted text (for example with datamarking) so the model treats it as data; it lowers injection success but is not a security boundary.

**Related:** [Prompt injection](#prompt-injection)

**Lesson:** [Securing AI systems](lessons/21-safety-security/01-ai-security.md)

### Standardization

**Plain English:** Rescaling a numeric feature so it has mean 0 and standard deviation 1, so features with large units don't dominate.

**Formal:** $z = (x - \mu)/\sigma$, with $\mu$ and $\sigma$ computed on the training split only.

**Example:** With training mean 5 and standard deviation 2, a value of 9 becomes 2.

**Related:** [Feature](#feature), [Data leakage](#data-leakage)

**Lesson:** [Feature engineering](lessons/24-applied-ml/01-feature-engineering.md)

### Stationarity

**Plain English:** A series is stationary when its average level, spread, and pattern of dependence don't change over time.

**Formal:** Weak stationarity: $E[y_t] = \mu$, $\text{Var}(y_t) = \sigma^2$, and $\text{Cov}(y_t, y_{t-k})$ depends only on $k$.

**Example:** Daily demand growing 0.05 per day is not stationary; its day-over-day differences are.

**Related:** [Autocorrelation](#autocorrelation), [Walk-forward validation](#walk-forward-validation)

**Lesson:** [Time series](lessons/24-applied-ml/04-time-series.md)

### Statistical power

**Plain English:** The probability that an experiment detects an effect when the effect is real.

**Formal:** Power $= 1 - \beta = P(\text{reject } H_0 \mid \text{true effect} = \delta)$; conventionally targeted at 0.8.

**Example:** With 14,749 users per arm and a 10% baseline, power is 0.8 for a true 10% relative lift but only about 0.29 for a 5% lift.

**Related:** [Minimum detectable effect](#minimum-detectable-effect), [P-value](#p-value)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Subgradient

**Plain English:** A stand-in for the gradient at a corner of a function that is not smooth, such as the hinge loss at $y f(x) = 1$; any slope between the left and right slopes works.

**Example:** For $|x|$ at $x = 0$, any value in $[-1, 1]$ is a subgradient.

**Lesson:** [Support vector machines](lessons/05-instance-and-probabilistic/03-support-vector-machines.md)

### Successive halving

**Plain English:** Train many configurations briefly, keep the best fraction, train those longer, and repeat; Hyperband runs several such brackets.

**Formal:** With $N$ configs, starting budget $b_0$, and factor $\eta$, round $r$ runs $N/\eta^r$ configs at budget $b_0\eta^r$, costing $N b_0$ per round.

**Example:** 27 configs at 1, 3, and 9 epochs cost 81 epochs, versus 243 to train all for 9.

**Related:** [Random search](#random-search), [Hyperparameter](#hyperparameter)

**Lesson:** [Hyperparameter tuning](lessons/24-applied-ml/03-hyperparameter-tuning.md)

### SwiGLU

**Plain English:** A gated feed-forward layer: one projection, passed through a Swish activation, multiplies a second projection elementwise before a down projection.

**Formal:** $\text{SwiGLU}(x) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\big)$: three matrices, $3hf$ parameters. GeGLU uses GELU instead of SiLU.

**Example:** Llama 3 8B: $3 \cdot 4{,}096 \cdot 14{,}336 = 176{,}160{,}768$ MLP parameters per layer.

**Related:** [Activation](#activation)

**Lesson:** [Open models: architecture walkthroughs](lessons/15-llms/07-open-models-architecture-walkthroughs.md)

### System prompt

**Plain English:** Standing instructions (role, rules, output format) sent with every call before the conversation.

**Formal:** The leading message(s) of the context, typically held fixed so they form a cacheable prefix; their tokens are billed on every call.

**Example:** A 600-token system prompt over 200,000 calls a day is 120 million input tokens a day.

**Related:** [Prefix caching](#prefix-caching), [Prompt injection](#prompt-injection)

**Lesson:** [Prompt engineering](lessons/28-prompting-and-context/01-prompt-engineering.md)

### Target encoding

**Plain English:** Replacing each category with the average label for that category, smoothed toward the overall average.

**Formal:** $\text{enc}(c) = (n_c \bar y_c + m \bar y)/(n_c + m)$; fit out of fold on training rows to avoid leaking each row's own label.

**Example:** A category with 2 rows, both positive, global rate 0.3, $m = 10$: $(2 + 3)/12 = 0.417$.

**Related:** [One-hot encoding](#one-hot-encoding), [Data leakage](#data-leakage)

**Lesson:** [Feature engineering](lessons/24-applied-ml/01-feature-engineering.md)

### Target network

**Plain English:** A frozen copy of a value network used only to compute training labels, so the labels do not shift with every gradient step.

**Formal:** Weights $\theta^-$ used in the TD target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$; updated by a hard copy $\theta^- \leftarrow \theta$ every $C$ steps or a soft update $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$.

**Example:** Atari DQN copied the weights every 10,000 steps; without a target network the Q estimate in the lesson's GridWorld swings to −43 against a true value of −5.85.

**Related:** [Deep Q-network (DQN)](#deep-q-network-dqn), [Deadly triad](#deadly-triad)

**Lesson:** [Deep Q-networks (DQN)](lessons/26-reinforcement-learning/03-dqn.md)

### Temperature

**Plain English:** A decoding knob that makes a model's choices more predictable (low) or more varied (high).

**Formal:** Logits are divided by $T$ before softmax; $T < 1$ sharpens, $T > 1$ flattens.

**Example:** Logits $[2, 1, 0]$: top probability 0.665 at $T = 1$, 0.867 at $T = 0.5$.

**Related:** [Softmax](#softmax)

**Lesson:** [Decoding](lessons/15-llms/02-decoding.md)

### Temporal-difference learning

**Plain English:** Learning a value estimate after every step by nudging it toward the reward plus your own estimate of the next state, without waiting for the episode to end.

**Formal:** $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$; the bracket is the TD error $\delta$.

**Example:** Q-learning, SARSA, DQN, and the critic in PPO all use TD errors.

**Related:** [Q-learning](#q-learning), [SARSA](#sarsa), [Bellman equation](#bellman-equation)

**Lesson:** [Q-learning and SARSA](lessons/26-reinforcement-learning/02-q-learning-sarsa.md)

### Tensor

**Plain English:** An n-dimensional array of numbers with a shape, a data type, and a device (CPU or GPU): the basic data structure of deep-learning frameworks.

**Formal:** A tensor of shape $[B, d]$ holds $B \times d$ values; operations broadcast trailing dimensions of size 1 or matching size.

**Example:** A batch of 32 RGB images of 224×224 pixels is a tensor of shape $[32, 3, 224, 224]$.

**Related:** [Autograd](#autograd), [Batch](#batch)

**Lesson:** [Intro to PyTorch](lessons/10-neural-networks/03-intro-to-pytorch.md)

### Tensor parallelism

**Plain English:** Splitting each weight matrix across GPUs so they share every layer's work.

**Formal:** Column-split outputs are concatenated; row-split outputs are summed with an all-reduce. Megatron-style MLPs column-split the first matrix and row-split the second, needing one all-reduce per block. Communicates every layer, so usually kept within one server.

**Example:** 8-way tensor parallelism cuts a 7B-style model's 107.8 GB of training state to 13.5 GB per GPU.

**Related:** [Pipeline parallelism](#pipeline-parallelism), [Data parallelism](#data-parallelism)

**Lesson:** [Scaling and distributed training](lessons/27-training-at-scale/04-scaling-distributed.md)

### TF-IDF

**Plain English:** A word weight that is high when a word is frequent in a document but rare across the corpus.

**Formal:** $\text{TF}(t,d)\times\ln(N/\text{DF}(t))$.

**Example:** 3 occurrences of a word found in 5 of 100 documents: $3 \times \ln 20 = 8.99$.

**Related:** [Cosine similarity](#cosine-similarity), [Embedding](#embedding)

**Lesson:** [From text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md)

### Time to first token

**Plain English:** How long a user waits from sending a request until the first word of the answer appears.

**Formal:** TTFT = queueing time + prefill time. Paired with TPOT (time per output token, or inter-token latency): end-to-end latency = TTFT + TPOT × (output tokens − 1).

**Example:** TTFT 200 ms and TPOT 25 ms for 300 tokens gives 7.675 s end to end.

**Related:** [Prefill](#prefill), [Inference](#inference)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Token

**Plain English:** The unit of text a language model actually reads and writes: often a word piece.

**Formal:** An element of the tokenizer's vocabulary, mapped to an integer ID and then to an embedding.

**Example:** "unbelievable" might be split into "un", "believ", "able".

**Related:** [Context window](#context-window), [Embedding](#embedding)

**Lesson:** [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md)

### Tool calling

**Plain English:** A model asking the application to run a function (search, look up an order, compute) by emitting a structured request.

**Formal:** The model outputs a function name and arguments matching a schema; the application validates, authorizes, executes, and returns the result.

**Example:** `get_order_status(order_id=123)`, executed only if the user owns order 123.

**Related:** [Agent](#agent), [Prompt injection](#prompt-injection)

**Lesson:** [Production architecture](lessons/20-production-ai/01-production-architecture.md)

### Tool poisoning

**Plain English:** Hidden instructions placed in a tool or MCP server description, which the model reads as trusted context.

**Related:** [Prompt injection](#prompt-injection), [Tool calling](#tool-calling)

**Lesson:** [Securing AI systems](lessons/21-safety-security/01-ai-security.md)

### Topic model

**Plain English:** An unsupervised model that finds recurring themes (groups of words that occur together) in a collection of documents.

**Formal:** LDA models each document as a mixture $\theta_d$ over $K$ topics and each topic as a distribution $\phi_k$ over the vocabulary; inference recovers $\theta$ $[D, K]$ and $\phi$ $[K, V]$ from word counts.

**Example:** A news article might be 60% "finance" and 40% "AI" topics.

**Related:** [TF-IDF](#tf-idf), [Embedding](#embedding)

**Lesson:** [Topic modeling](lessons/09-classical-nlp/04-topic-modeling.md)

### Training

**Plain English:** Fitting a model's parameters to data by minimizing a loss.

**Formal:** $\theta^* = \arg\min_\theta \frac{1}{n}\sum_i\ell(f_\theta(x_i), y_i)$, usually by gradient-based optimization.

**Example:** Running gradient descent for 2,000 steps to fit a regression line.

**Related:** [Inference](#inference), [Loss](#loss)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Training-serving skew

**Plain English:** Any difference between the data a model sees during training and the data it gets in production, caused by the pipeline rather than by the world changing.

**Formal:** $P_{\text{train}}(x) \ne P_{\text{serve}}(x)$ for the same underlying events, due to different feature code, defaults, freshness, or preprocessing between the offline and online paths.

**Example:** A counter computed hourly in the training pipeline but read as 0 when the online cache misses makes the model wrong on exactly the requests where the cache fails.

**Related:** [Feature store](#feature-store), [Drift](#drift), [Data leakage](#data-leakage)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

### Transformer

**Plain English:** The neural architecture behind modern LLMs: stacked blocks of attention and feed-forward layers with residual connections.

**Formal:** Blocks of $x + \text{MHA}(\text{LN}(x))$ followed by $x + \text{FFN}(\text{LN}(x))$, operating on token embeddings plus positional information.

**Example:** GPT-2 small has 12 blocks of width 768 and 124M parameters.

**Related:** [Attention](#attention), [Token](#token)

**Lesson:** [The transformer architecture](lessons/14-transformers/02-transformer-architecture.md)

### Two-tower model

**Plain English:** A retrieval model with one network for users (or queries) and one for items, scoring a pair by the dot product of their vectors.

**Formal:** $s(u, i) = f_\theta(u) \cdot g_\phi(i)$ with $f_\theta(u), g_\phi(i) \in \mathbb{R}^d$; item vectors are precomputed and searched with an approximate nearest neighbor index.

**Example:** A feed retrieves 1,000 of 10M videos in milliseconds by computing one 64-dimensional user vector and querying an ANN index of item vectors.

**Related:** [Embedding](#embedding), [Cold start](#cold-start)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

### Vision transformer

**Plain English:** A transformer applied to an image cut into square patches, each patch treated like a word token.

**Formal:** Patches of $P \times P$ pixels are linearly embedded (a stride-$P$ convolution), a [CLS] token and learned position embeddings are added, and a standard pre-norm transformer encoder processes the $HW/P^2 + 1$ tokens.

**Example:** ViT-B/16 on a 224-pixel image uses 196 patch tokens plus [CLS] and has 86.6M parameters.

**Related:** [Transformer](#transformer), [Attention](#attention), [Embedding](#embedding)

**Lesson:** [Vision transformers](lessons/25-computer-vision/03-vision-transformers.md)

### Vision-language model

**Plain English:** A language model that can read images: a vision encoder's features are translated into the LLM's token space and placed in the prompt.

**Formal:** Vision encoder (usually a CLIP or SigLIP ViT) → projector (often a two-layer MLP) → decoder LLM; projected patch features replace `<image>` placeholder tokens.

**Example:** A 336-pixel image with 14-pixel patches adds 576 tokens to the prompt.

**Related:** [Transformer](#transformer), [Token](#token), [Hallucination](#hallucination)

**Lesson:** [Vision-language models](lessons/25-computer-vision/06-vision-language-models.md)

### Viterbi algorithm

**Plain English:** Dynamic programming that finds the highest-scoring sequence of labels without trying every sequence.

**Formal:** $\delta_i(k) = e_i(k) + \max_j[\delta_{i-1}(j) + A(j, k)]$ with back-pointers; cost $O(nK^2)$ for $n$ tokens and $K$ tags.

**Example:** With "New" P(O)=0.6, P(B-LOC)=0.4 and "York" P(I-LOC)=0.5, Viterbi picks B-LOC I-LOC (0.20) over the invalid greedy O I-LOC.

**Related:** [BIO tagging](#bio-tagging)

**Lesson:** [Named entity recognition](lessons/09-classical-nlp/03-named-entity-recognition.md)

### Walk-forward validation

**Plain English:** Validating a forecaster by training on the past, testing on the next block, then moving forward and repeating.

**Formal:** Fold $k$ trains on rows before $s_k$ and tests on $[s_k, s_k + h)$, with $s_1 < s_2 < \dots$; expanding or fixed-length (sliding) training windows.

**Example:** scikit-learn's `TimeSeriesSplit(n_splits=5, test_size=60)`.

**Related:** [Data leakage](#data-leakage), [Stationarity](#stationarity)

**Lesson:** [Time series](lessons/24-applied-ml/04-time-series.md)

### Weight initialization

**Plain English:** Choosing the random scale of starting weights so activations neither shrink to zero nor blow up as they pass through many layers.

**Formal:** Xavier/LeCun: $\text{Var}(w) = 1/\text{fan\_in}$ (or $2/(\text{fan\_in}+\text{fan\_out})$) for tanh; He: $\text{Var}(w) = 2/\text{fan\_in}$ for ReLU.

**Example:** At width 256, a fixed std of 0.01 leaves activations about 1e-8 after 10 tanh layers; He init keeps ReLU activations near 1 after 20.

**Lesson:** [Training and regularization](lessons/12-training-regularization/01-training-and-regularization.md)

### Word error rate

**Plain English:** The standard speech-recognition metric: the fraction of reference words you'd have to fix to turn the transcript into the reference.

**Formal:** $\text{WER} = (S + D + I)/N$, word-level edit distance divided by the number of reference words, computed after text normalization; it can exceed 1.

**Example:** Reference "turn on the kitchen lights", hypothesis "turn the kitchen light on": 1 substitution, 1 deletion, 1 insertion, WER $= 3/5 = 0.6$.

**Related:** [CTC](#ctc)

**Lesson:** [Speech recognition: CTC, transducers, and attention](lessons/29-speech-and-audio/02-speech-recognition-asr.md)

### ZeRO

**Plain English:** A way to stop every data-parallel GPU from storing duplicate copies of optimizer state, gradients, and weights.

**Formal:** Zero Redundancy Optimizer (Rajbhandari et al., 2020). Per-GPU memory with mixed-precision AdamW: stage 1 $4P + 12P/N$, stage 2 $2P + 14P/N$, stage 3 $16P/N$, versus $16P$ for DDP. Stages 1 and 2 send the same bytes as DDP; stage 3 about 1.5 times as much.

**Example:** A 7B-style model on 8 GPUs: 108.6 GB per GPU under DDP, 37.9 GB under ZeRO-1, 14.3 GB under ZeRO-3 (including activations).

**Related:** [FSDP](#fsdp), [Reduce-scatter](#reduce-scatter), [Data parallelism](#data-parallelism)

**Lesson:** [ZeRO and FSDP](lessons/27-training-at-scale/05-zero-parameter-sharding.md)

### Zero-shot classification

**Plain English:** Classifying into categories the model was never trained on, by comparing the input with text descriptions of each category.

**Formal:** With a shared image-text embedding space, $\hat{y} = \arg\max_k \cos(f(x), g(\text{prompt}_k))$.

**Example:** CLIP labels a photo by embedding "a photo of a cat", "a photo of a dog", ... and picking the closest.

**Related:** [Embedding](#embedding), [Cosine similarity](#cosine-similarity)

**Lesson:** [CLIP and contrastive image-text pretraining](lessons/25-computer-vision/05-clip-contrastive-pretraining.md)
