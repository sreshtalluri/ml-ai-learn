# Glossary

Plain-English definitions first, then formal ones, with a small example and the lesson that teaches each term. On the [website](https://sreshtalluri.github.io/ml-ai-learn/glossary/) this page is searchable, and linked terms show their definition on hover inside lessons.

### Activation

**Plain English:** The nonlinear function a neuron applies to its weighted sum, so the network can bend its decision boundaries.

**Formal:** $a = f(z)$ with $z = w\cdot x + b$; common $f$ are ReLU, GELU, sigmoid, tanh.

**Example:** ReLU turns $z = -1.5$ into 0 and $z = 2$ into 2.

**Related:** [Logit](#logit), [Softmax](#softmax)

**Lesson:** [The forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md)

### Agent

**Plain English:** A system that uses a model in a loop to plan, call tools, observe results, and decide what to do next.

**Formal:** A controller that alternates model calls and tool executions until a stop condition, under step, cost, and permission limits.

**Example:** A support agent looks up an order, checks the refund policy, and drafts a reply for human approval.

**Related:** [Tool calling](#tool-calling), [Prompt injection](#prompt-injection)

**Lesson:** [Production architecture](lessons/20-production-ai/01-production-architecture.md)

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

### Bias-variance trade-off

**Plain English:** Simple models are consistently wrong (bias); flexible models change a lot with the training data (variance); the best model balances both.

**Formal:** Expected squared error $= \text{bias}^2 + \text{variance} + \sigma^2$.

**Example:** A degree-1 polynomial underfits a sine curve; degree 15 overfits it.

**Related:** [Overfitting](#overfitting), [Regularization](#regularization)

**Lesson:** [Overfitting and bias-variance](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md)

### BM25

**Plain English:** A keyword-ranking formula that scores documents by how often they contain the query's words, giving more weight to rare words and less to very long documents.

**Formal:** $\text{BM25}(q, D) = \sum_{t \in q} \text{IDF}(t)\,\frac{f(t,D)(k_1+1)}{f(t,D) + k_1(1 - b + b\,|D|/\text{avgdl})}$, typically $k_1 \approx 1.2$, $b \approx 0.75$.

**Example:** A query for an error code like "E1234" is matched exactly by BM25 even when a dense embedding treats it as noise.

**Related:** [TF-IDF](#tf-idf), [Reciprocal rank fusion](#reciprocal-rank-fusion)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Calibration

**Plain English:** Whether a model's predicted probabilities match how often things actually happen.

**Formal:** A model is calibrated if $P(y = 1 \mid \hat{p} = p) = p$ for all $p$; measured with reliability diagrams or the Brier score.

**Example:** Of all emails scored 0.8 spam, about 80% should really be spam.

**Related:** [Logit](#logit), [Hallucination](#hallucination)

**Lesson:** [Classification metrics](lessons/04-classification/02-classification-metrics.md)

### Catastrophic forgetting

**Plain English:** When training a model on a new, narrow task makes it lose skills it had before.

**Formal:** Performance on the original distribution drops after fine-tuning because gradient updates for the new objective overwrite weights the old behavior depended on.

**Example:** A model fine-tuned for many epochs on support tickets gets better at tickets but worse at general reasoning and safety refusals.

**Related:** [Fine-tuning](#fine-tuning), [LoRA](#lora)

**Lesson:** [Fine-tuning in practice](lessons/15-llms/04-fine-tuning-in-practice.md)

### Checkpoint

**Plain English:** A saved snapshot of a model's weights (and often optimizer state) during or after training.

**Formal:** A serialized copy of $\theta$ at step $t$, used to resume training, evaluate, or deploy.

**Example:** Early stopping deploys the checkpoint with the lowest validation loss, not the last one.

**Related:** [Epoch](#epoch)

**Lesson:** [Training and regularization](lessons/12-training-regularization/01-training-and-regularization.md)

### Chunking

**Plain English:** Splitting documents into smaller pieces that can be retrieved individually.

**Formal:** Partitioning text into segments of size $S$ with overlap $O$, giving about $\lceil (N - O)/(S - O)\rceil$ chunks.

**Example:** A 1,000-word document with 200-word chunks and 50-word overlap gives 7 chunks.

**Related:** [RAG](#rag), [Embedding](#embedding)

**Lesson:** [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md)

### Cold start

**Plain English:** The problem of making good predictions for a new user or item that has no interaction history yet.

**Formal:** For an entity with no logged interactions, collaborative signals are undefined, so the model must rely on content and context features, priors such as popularity, or deliberate exploration.

**Example:** A video uploaded a minute ago has no views; the item tower embeds its title and frames so it can still be retrieved, and re-ranking gives it a small freshness boost.

**Related:** [Two-tower model](#two-tower-model), [Position bias](#position-bias)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

### Constrained decoding

**Plain English:** Forcing a model's output to follow a format, such as a JSON schema, by blocking any next token that would break it.

**Formal:** At each step, tokens outside the allowed set $A_t$ (from a grammar compiled from the schema) get probability 0 and the rest are renormalized: $\tilde p(y_t) \propto p(y_t)\,\mathbb{1}[y_t \in A_t]$.

**Example:** A tool call generated under the schema always parses, though `flight_id` may still be the wrong flight.

**Related:** [Tool calling](#tool-calling), [Temperature](#temperature)

**Lesson:** [Tool use and agents](lessons/19-agents/01-tool-use-and-agents.md)

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

### Cosine similarity

**Plain English:** How closely two vectors point in the same direction, ignoring their lengths.

**Formal:** $\cos(a, b) = \frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$, between −1 and 1.

**Example:** $[1, 2, 0]$ and $[2, 1, 1]$ have cosine 0.730.

**Related:** [Embedding](#embedding), [TF-IDF](#tf-idf)

**Lesson:** [From text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md)

### Cross-entropy

**Plain English:** A loss that punishes a model for giving low probability to the correct answer.

**Formal:** $-\ln p_{\text{true}}$ for one example; binary form $-[y\ln p + (1-y)\ln(1-p)]$.

**Example:** Probability 0.9 on the true class costs 0.105; probability 0.01 costs 4.6.

**Related:** [Loss](#loss), [Perplexity](#perplexity)

**Lesson:** [Logistic regression](lessons/04-classification/01-logistic-regression.md)

### CUPED

**Plain English:** A variance-reduction trick for A/B tests that subtracts the part of each user's outcome that was already predictable from their behavior before the experiment.

**Formal:** $Y^{\text{cuped}} = Y - \theta(X - \bar{X})$ with $\theta = \operatorname{Cov}(X, Y)/\operatorname{Var}(X)$, where $X$ is a pre-experiment covariate; the variance shrinks by a factor $1 - \rho^2$ and the estimate stays unbiased.

**Example:** With $\rho = 0.5$ between last month's and this month's sessions, variance drops 25%, so a test needing 14,749 users per arm needs about 11,062.

**Related:** [Statistical power](#statistical-power), [Minimum detectable effect](#minimum-detectable-effect)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Data leakage

**Plain English:** When training uses information that won't be available at prediction time, making results look better than they are.

**Formal:** Any dependence of training features or preprocessing on test data, future data, or the target itself.

**Example:** Fitting a scaler on all data before splitting, or including "refund issued" when predicting fraud.

**Related:** [Baseline](#baseline), [Generalization](#generalization)

**Lesson:** [Workflow, splits, and leakage](lessons/02-ml-workflow/01-ml-workflow.md)

### Distillation

**Plain English:** Training a small model to imitate a large one, to get similar quality more cheaply.

**Formal:** Minimizing the divergence between the student's outputs and the teacher's (soft) outputs, often alongside the true labels.

**Example:** A small classifier trained on a large LLM's labels serves high-volume traffic at a fraction of the cost.

**Related:** [Quantization](#quantization), [Inference](#inference)

**Lesson:** [Reliability, cost, and observability](lessons/20-production-ai/02-reliability-cost-and-observability.md)

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

### Epoch

**Plain English:** One full pass over the training data.

**Formal:** $\lceil n/|B|\rceil$ optimization steps for $n$ examples and batch size $|B|$.

**Example:** Validation loss is usually checked once per epoch for early stopping.

**Related:** [Batch](#batch), [Checkpoint](#checkpoint)

**Lesson:** [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md)

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

### Generalization

**Plain English:** How well a model performs on new data it never saw during training.

**Formal:** Expected loss on the data distribution, estimated by held-out data that played no part in building the model.

**Example:** A model with 99% training accuracy and 70% test accuracy generalizes poorly.

**Related:** [Overfitting](#overfitting), [Data leakage](#data-leakage)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Gradient

**Plain English:** The direction and steepness of the loss surface: which way each parameter should move to increase the loss fastest.

**Formal:** $\nabla_\theta L = [\partial L/\partial\theta_1, \dots, \partial L/\partial\theta_P]$.

**Example:** For $L = x^2 + 2y^2$ at $(1, 1)$, $\nabla L = (2, 4)$.

**Related:** [Gradient descent](#gradient-descent), [Backpropagation](#backpropagation)

**Lesson:** [Calculus for ML](lessons/00-foundations/02-calculus-for-ml.md)

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

### Hallucination

**Plain English:** A fluent, confident statement that isn't supported by facts or the provided sources.

**Formal:** Generated content inconsistent with the source context or the world, arising because the model optimizes likely text rather than truth.

**Example:** Inventing a refund policy detail that appears in no document.

**Related:** [Grounding](#grounding), [Calibration](#calibration)

**Lesson:** [Adapting LLMs](lessons/15-llms/03-adapting-llms.md)

### HNSW

**Plain English:** Hierarchical navigable small world: a vector index that links each vector to its near neighbors in a layered graph and searches by hopping toward the query.

**Formal:** Each node keeps up to $M$ links per upper layer and $2M$ at layer 0; search descends greedily from the top layer, then runs best-first search with a candidate list of size `efSearch`.

**Example:** With $M = 32$ and 4-byte IDs, the bottom-layer links cost about 256 bytes per vector, 256 MB per million vectors.

**Related:** [Approximate nearest neighbor search](#approximate-nearest-neighbor-search), [Embedding](#embedding)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Hyperparameter

**Plain English:** A setting you choose before training that controls how learning happens.

**Formal:** A value fixed outside the optimization of $\theta$, usually selected on a validation set.

**Example:** Learning rate, tree depth, $K$ in K-nearest neighbors.

**Related:** [Parameter](#parameter)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Inference

**Plain English:** Running a trained model to make predictions on new inputs.

**Formal:** Computing $f_{\theta^*}(x_{\text{new}})$ with fixed parameters; for LLMs, generating tokens from a prompt.

**Example:** Scoring a new transaction for fraud in 20 milliseconds.

**Related:** [Training](#training), [Quantization](#quantization)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

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

### Learning rate

**Plain English:** How big a step the optimizer takes each update.

**Formal:** The scalar $\eta$ in $\theta \leftarrow \theta - \eta\nabla L$; often scheduled (warmup, decay).

**Example:** Too high diverges (loss explodes); too low trains very slowly.

**Related:** [Gradient descent](#gradient-descent), [Hyperparameter](#hyperparameter)

**Lesson:** [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md)

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

### Metric

**Plain English:** How people judge a model's performance, which can differ from the loss used for training.

**Formal:** Any evaluation function of predictions and labels, possibly non-differentiable and threshold-dependent.

**Example:** RMSE in dollars, recall at a review budget, p95 latency, cost per request.

**Related:** [Loss](#loss), [Baseline](#baseline)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Minimum detectable effect

**Plain English:** The smallest true improvement an experiment is designed to detect reliably.

**Formal:** The effect size $\delta$ at which the test has the chosen power $1 - \beta$ at significance $\alpha$; the sample size per arm scales as $1/\delta^2$.

**Example:** Detecting 10% → 11% conversion (a 10% relative MDE) at $\alpha = 0.05$ and power 0.8 takes 14,749 users per arm; halving the MDE takes about four times as many.

**Related:** [Statistical power](#statistical-power), [P-value](#p-value)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Mixture of experts

**Plain English:** A model whose feed-forward layers are split into many "experts", with a router sending each token to only a few of them, so it has many parameters but uses few per token.

**Formal:** Each MoE layer has $E$ expert networks and a router that picks the top $k$ per token. Total parameters set memory; active parameters (shared plus $k$ experts) set compute per token.

**Example:** 4B shared + 16 experts × 2B with top-2 routing: 36B total (72 GB in BF16), 8B active per token.

**Related:** [Parameter](#parameter), [Inference](#inference)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Model Context Protocol (MCP)

**Plain English:** An open standard for connecting AI applications to tools and data, so one integration works with many apps.

**Formal:** MCP clients inside a host application talk JSON-RPC (over stdio or HTTP) to MCP servers, which expose tools (callable functions), resources (readable data), and prompts (templates).

**Example:** An issue-tracker MCP server lets any compliant IDE or agent search and create tickets without custom glue code.

**Related:** [Tool calling](#tool-calling), [Agent](#agent), [Prompt injection](#prompt-injection)

**Lesson:** [Tool use and agents](lessons/19-agents/01-tool-use-and-agents.md)

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

### pass^k

**Plain English:** The chance an agent gets a task right on every one of $k$ independent tries, a measure of consistency.

**Formal:** For per-trial success rate $s$, $\text{pass}^k = s^k$, versus $\text{pass@}k = 1 - (1-s)^k$ (at least one success).

**Example:** With $s = 0.7$, pass^3 $= 0.343$ while pass@3 $= 0.973$.

**Related:** [Agent](#agent), [Metric](#metric)

**Lesson:** [Building reliable agents](lessons/19-agents/02-reliable-agents.md)

### Perplexity

**Plain English:** How "surprised" a language model is by text, roughly the number of tokens it's choosing between.

**Formal:** $\exp$ of the mean next-token cross-entropy.

**Example:** Mean loss 1.25 gives perplexity 3.49; uniform guessing over 50,000 tokens gives 50,000.

**Related:** [Cross-entropy](#cross-entropy), [Token](#token)

**Lesson:** [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md)

### Position bias

**Plain English:** Users interact with items partly because of where they are shown, so top slots get more clicks regardless of quality.

**Formal:** $P(\text{click} \mid \text{item}, \text{position}) \approx P(\text{examined} \mid \text{position}) \cdot P(\text{relevant} \mid \text{item})$; training on raw clicks confounds the two factors.

**Example:** A ranker trained on raw clicks learns to promote whatever the previous model put first, creating a feedback loop; adding position as a feature fixed at serving time, or inverse propensity weighting, corrects for it.

**Related:** [Cold start](#cold-start), [Calibration](#calibration)

**Lesson:** [ML system design](lessons/22-ml-system-design/01-ml-system-design.md)

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

### Prompt injection

**Plain English:** Text that tries to override a model's instructions, typed by a user or hidden in content the model reads.

**Formal:** Adversarial input exploiting the model's inability to separate trusted instructions from untrusted data; indirect injection arrives via retrieved documents, pages, emails, or tool outputs.

**Example:** A web page containing hidden text telling the assistant to email the user's invoices elsewhere.

**Related:** [Agent](#agent), [Tool calling](#tool-calling)

**Lesson:** [Securing AI systems](lessons/21-safety-security/01-ai-security.md)

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

### RAG

**Plain English:** Retrieval-augmented generation: look up relevant documents at query time and give them to the model to answer from.

**Formal:** A pipeline of ingestion, chunking, indexing, retrieval, optional reranking, context construction, and grounded generation; it changes the model's context, not its weights.

**Example:** A support bot retrieves the current refund policy and cites it.

**Related:** [Grounding](#grounding), [Fine-tuning](#fine-tuning)

**Lesson:** [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md)

### ReAct

**Plain English:** An agent pattern where the model alternates short reasoning notes with tool calls and reads each result before deciding the next step.

**Formal:** The trajectory interleaves thought, action, and observation: $(\tau_1, a_1, o_1, \tau_2, a_2, o_2, \dots)$, each generated conditioned on everything before it.

**Example:** "UA412 is cheapest; check refundability" → `get_fare_rules(UA412)` → `{"refundable": false}` → "check AS330".

**Related:** [Agent](#agent), [Tool calling](#tool-calling)

**Lesson:** [Tool use and agents](lessons/19-agents/01-tool-use-and-agents.md)

### Reciprocal rank fusion

**Plain English:** A way to merge ranked lists from different search methods using only each document's rank, not its score.

**Formal:** $\text{RRF}(d) = \sum_j \frac{1}{c + \text{rank}_j(d)}$, commonly $c = 60$.

**Example:** A document ranked 1st by dense search and 3rd by BM25 scores $1/61 + 1/63 = 0.0323$.

**Related:** [BM25](#bm25), [RAG](#rag)

**Lesson:** [Vector search](lessons/16-rag/02-vector-search.md)

### Regularization

**Plain English:** Anything that discourages a model from memorizing, so it generalizes better.

**Formal:** Penalties ($\lambda\lVert w\rVert^2$, $\lambda\lVert w\rVert_1$) or procedures (dropout, early stopping, augmentation) that reduce variance.

**Example:** Ridge regression shrinks the slope from 1.5 to 0.6 with $\lambda = 1$ in the course example.

**Related:** [Overfitting](#overfitting)

**Lesson:** [Regularization and regression metrics](lessons/03-regression/02-regularization-and-regression-metrics.md)

### Sample ratio mismatch

**Plain English:** When the number of users in each arm of an experiment differs from the planned split by more than chance allows, a sign that assignment or logging is broken.

**Formal:** Tested with a chi-square goodness-of-fit test on arm counts against the designed allocation; a tiny p-value (often below 0.001) invalidates the experiment's readout.

**Example:** 50,600 versus 49,400 users in a 50/50 test gives $\chi^2 = 14.4$ and $p \approx 0.00015$.

**Related:** [P-value](#p-value)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Softmax

**Plain English:** Turns a list of scores into probabilities that are all positive and add up to 1.

**Formal:** $p_i = e^{z_i/T}/\sum_j e^{z_j/T}$ with temperature $T$.

**Example:** Logits $[2, 1, 0.1]$ become $[0.659, 0.242, 0.099]$.

**Related:** [Logit](#logit), [Temperature](#temperature)

**Lesson:** [Logistic regression](lessons/04-classification/01-logistic-regression.md)

### Speculative decoding

**Plain English:** A cheap draft model guesses several next tokens, and the large model checks them all in one pass, keeping the ones it agrees with.

**Formal:** With draft length $\gamma$ and per-token acceptance rate $\alpha$, the expected tokens per target pass is $(1 - \alpha^{\gamma+1})/(1 - \alpha)$. With rejection-sampling verification, outputs follow the target model's distribution exactly.

**Example:** $\alpha = 0.8$, $\gamma = 4$: 3.36 tokens per target pass; with draft cost 0.05 per token, a 2.8× speedup.

**Related:** [Inference](#inference), [Temperature](#temperature)

**Lesson:** [LLM inference](lessons/18-llm-inference/01-llm-inference.md)

### Statistical power

**Plain English:** The probability that an experiment detects an effect when the effect is real.

**Formal:** Power $= 1 - \beta = P(\text{reject } H_0 \mid \text{true effect} = \delta)$; conventionally targeted at 0.8.

**Example:** With 14,749 users per arm and a 10% baseline, power is 0.8 for a true 10% relative lift but only about 0.29 for a 5% lift.

**Related:** [Minimum detectable effect](#minimum-detectable-effect), [P-value](#p-value)

**Lesson:** [Experimentation and A/B testing](lessons/22-ml-system-design/02-experimentation-and-ab-testing.md)

### Temperature

**Plain English:** A decoding knob that makes a model's choices more predictable (low) or more varied (high).

**Formal:** Logits are divided by $T$ before softmax; $T < 1$ sharpens, $T > 1$ flattens.

**Example:** Logits $[2, 1, 0]$: top probability 0.665 at $T = 1$, 0.867 at $T = 0.5$.

**Related:** [Softmax](#softmax)

**Lesson:** [Decoding](lessons/15-llms/02-decoding.md)

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
