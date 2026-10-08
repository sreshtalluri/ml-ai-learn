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

**Lesson:** [Production architecture](lessons/18-production-ai/01-production-architecture.md)

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

### Calibration

**Plain English:** Whether a model's predicted probabilities match how often things actually happen.

**Formal:** A model is calibrated if $P(y = 1 \mid \hat{p} = p) = p$ for all $p$; measured with reliability diagrams or the Brier score.

**Example:** Of all emails scored 0.8 spam, about 80% should really be spam.

**Related:** [Logit](#logit), [Hallucination](#hallucination)

**Lesson:** [Classification metrics](lessons/04-classification/02-classification-metrics.md)

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

### Context window

**Plain English:** The maximum number of tokens a language model can consider at once, prompt and output combined.

**Formal:** The maximum sequence length $n_{\max}$ over which attention is computed.

**Example:** Anything beyond the window is invisible to the model unless retrieved and inserted.

**Related:** [Token](#token), [Attention](#attention)

**Lesson:** [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md)

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

**Lesson:** [Reliability, cost, and observability](lessons/18-production-ai/02-reliability-cost-and-observability.md)

### Drift

**Plain English:** When the data or the world changes after deployment, so a model's performance degrades.

**Formal:** Data drift is a change in $P(x)$; concept drift is a change in $P(y \mid x)$; prompt drift is a change in what users ask.

**Example:** A demand model trained before a price change starts under-predicting.

**Related:** [Generalization](#generalization), [Calibration](#calibration)

**Lesson:** [Reliability, cost, and observability](lessons/18-production-ai/02-reliability-cost-and-observability.md)

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

### Fine-tuning

**Plain English:** Continuing to train a pretrained model on task-specific data so it changes its behavior.

**Formal:** Further optimization of some or all pretrained weights (or added adapters such as LoRA) on a new dataset.

**Example:** Fine-tuning a model on support transcripts to adopt a consistent tone and format.

**Related:** [RAG](#rag), [LoRA](#lora)

**Lesson:** [Adapting LLMs](lessons/15-llms/03-adapting-llms.md)

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

### Hallucination

**Plain English:** A fluent, confident statement that isn't supported by facts or the provided sources.

**Formal:** Generated content inconsistent with the source context or the world, arising because the model optimizes likely text rather than truth.

**Example:** Inventing a refund policy detail that appears in no document.

**Related:** [Grounding](#grounding), [Calibration](#calibration)

**Lesson:** [Adapting LLMs](lessons/15-llms/03-adapting-llms.md)

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

### Overfitting

**Plain English:** When a model learns the training data's noise and does worse on new data.

**Formal:** A large gap between training and validation performance, typically from high variance.

**Example:** A degree-15 polynomial with training MSE 0.056 and validation MSE 0.52.

**Related:** [Regularization](#regularization), [Bias-variance trade-off](#bias-variance-trade-off)

**Lesson:** [Overfitting and bias-variance](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md)

### Parameter

**Plain English:** A number the model learns from data during training.

**Formal:** An entry of $\theta$, the vector of values the optimizer adjusts to minimize the loss $L(\theta)$.

**Example:** The slope $w$ and bias $b$ in $\hat{y} = wx + b$.

**Related:** [Hyperparameter](#hyperparameter), [Gradient](#gradient)

**Lesson:** [Linear regression](lessons/03-regression/01-linear-regression.md)

### Perplexity

**Plain English:** How "surprised" a language model is by text, roughly the number of tokens it's choosing between.

**Formal:** $\exp$ of the mean next-token cross-entropy.

**Example:** Mean loss 1.25 gives perplexity 3.49; uniform guessing over 50,000 tokens gives 50,000.

**Related:** [Cross-entropy](#cross-entropy), [Token](#token)

**Lesson:** [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md)

### Prompt injection

**Plain English:** Text that tries to override a model's instructions, typed by a user or hidden in content the model reads.

**Formal:** Adversarial input exploiting the model's inability to separate trusted instructions from untrusted data; indirect injection arrives via retrieved documents, pages, emails, or tool outputs.

**Example:** A web page containing hidden text telling the assistant to email the user's invoices elsewhere.

**Related:** [Agent](#agent), [Tool calling](#tool-calling)

**Lesson:** [Securing AI systems](lessons/19-safety-security/01-ai-security.md)

### Quantization

**Plain English:** Storing model weights with fewer bits to save memory and often speed up inference.

**Formal:** Mapping weights from 16 or 32-bit floats to 8 or 4-bit integers with scale factors; memory = parameters × bits / 8.

**Example:** A 7B model needs 14 GB in fp16 and 3.5 GB in int4.

**Related:** [Distillation](#distillation), [Inference](#inference)

**Lesson:** [Reliability, cost, and observability](lessons/18-production-ai/02-reliability-cost-and-observability.md)

### RAG

**Plain English:** Retrieval-augmented generation: look up relevant documents at query time and give them to the model to answer from.

**Formal:** A pipeline of ingestion, chunking, indexing, retrieval, optional reranking, context construction, and grounded generation; it changes the model's context, not its weights.

**Example:** A support bot retrieves the current refund policy and cites it.

**Related:** [Grounding](#grounding), [Fine-tuning](#fine-tuning)

**Lesson:** [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md)

### Regularization

**Plain English:** Anything that discourages a model from memorizing, so it generalizes better.

**Formal:** Penalties ($\lambda\lVert w\rVert^2$, $\lambda\lVert w\rVert_1$) or procedures (dropout, early stopping, augmentation) that reduce variance.

**Example:** Ridge regression shrinks the slope from 1.5 to 0.6 with $\lambda = 1$ in the course example.

**Related:** [Overfitting](#overfitting)

**Lesson:** [Regularization and regression metrics](lessons/03-regression/02-regularization-and-regression-metrics.md)

### Softmax

**Plain English:** Turns a list of scores into probabilities that are all positive and add up to 1.

**Formal:** $p_i = e^{z_i/T}/\sum_j e^{z_j/T}$ with temperature $T$.

**Example:** Logits $[2, 1, 0.1]$ become $[0.659, 0.242, 0.099]$.

**Related:** [Logit](#logit), [Temperature](#temperature)

**Lesson:** [Logistic regression](lessons/04-classification/01-logistic-regression.md)

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

**Lesson:** [Production architecture](lessons/18-production-ai/01-production-architecture.md)

### Training

**Plain English:** Fitting a model's parameters to data by minimizing a loss.

**Formal:** $\theta^* = \arg\min_\theta \frac{1}{n}\sum_i\ell(f_\theta(x_i), y_i)$, usually by gradient-based optimization.

**Example:** Running gradient descent for 2,000 steps to fit a regression line.

**Related:** [Inference](#inference), [Loss](#loss)

**Lesson:** [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md)

### Transformer

**Plain English:** The neural architecture behind modern LLMs: stacked blocks of attention and feed-forward layers with residual connections.

**Formal:** Blocks of $x + \text{MHA}(\text{LN}(x))$ followed by $x + \text{FFN}(\text{LN}(x))$, operating on token embeddings plus positional information.

**Example:** GPT-2 small has 12 blocks of width 768 and 124M parameters.

**Related:** [Attention](#attention), [Token](#token)

**Lesson:** [The transformer architecture](lessons/14-transformers/02-transformer-architecture.md)
