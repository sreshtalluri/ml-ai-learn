---
title: Important formulas
summary: The formulas worth remembering, with the lesson that derives each.
---

# Important formulas

| Concept | Formula | Lesson |
|---|---|---|
| Dot product | $x\cdot w = \sum_i x_i w_i$ | [Vectors](../lessons/00-foundations/01-vectors-and-matrices.md) |
| Linear model | $\hat{y} = w\cdot x + b$ | [Linear regression](../lessons/03-regression/01-linear-regression.md) |
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | [Linear regression](../lessons/03-regression/01-linear-regression.md) |
| MSE gradients | $\frac{2}{n}\sum(\hat{y}-y)x$, $\frac{2}{n}\sum(\hat{y}-y)$ | [Linear regression](../lessons/03-regression/01-linear-regression.md) |
| Ridge / lasso penalty | $\lambda\sum w^2$ / $\lambda\sum\lvert w\rvert$ | [Regularization](../lessons/03-regression/02-regularization-and-regression-metrics.md) |
| Sigmoid | $\sigma(z) = 1/(1 + e^{-z})$ | [Logistic regression](../lessons/04-classification/01-logistic-regression.md) |
| Binary cross-entropy | $-[y\ln p + (1-y)\ln(1-p)]$ | [Logistic regression](../lessons/04-classification/01-logistic-regression.md) |
| Softmax with temperature | $p_i = e^{z_i/T}/\sum_j e^{z_j/T}$ | [Decoding](../lessons/15-llms/02-decoding.md) |
| Precision, recall, F1 | $\frac{TP}{TP+FP}$, $\frac{TP}{TP+FN}$, $\frac{2PR}{P+R}$ | [Metrics](../lessons/04-classification/02-classification-metrics.md) |
| Bayes' rule | $P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)}$ | [Probability](../lessons/00-foundations/03-probability-and-statistics.md) |
| Euclidean distance | $\sqrt{\sum(q_j - x_j)^2}$ | [KNN](../lessons/05-instance-and-probabilistic/01-k-nearest-neighbors.md) |
| Gini impurity / entropy | $1 - \sum p_k^2$ / $-\sum p_k\log_2 p_k$ | [Decision trees](../lessons/06-trees-and-ensembles/01-decision-trees.md) |
| Boosting update | $F_m = F_{m-1} + \eta h_m$ | [Ensembles](../lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md) |
| K-means objective | $\sum_i\lVert x_i - \mu_{c(i)}\rVert^2$ | [K-means](../lessons/07-unsupervised/01-k-means.md) |
| PCA | $Cv = \lambda v$; ratio $\lambda_k/\sum\lambda$ | [PCA](../lessons/08-dimensionality-reduction/01-pca.md) |
| TF-IDF | $\text{TF}(t,d)\times\ln(N/\text{DF}(t))$ | [Text to vectors](../lessons/09-classical-nlp/01-text-to-vectors.md) |
| Cosine similarity | $\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$ | [Text to vectors](../lessons/09-classical-nlp/01-text-to-vectors.md) |
| Layer | $a = f(Wx + b)$ | [Forward pass](../lessons/10-neural-networks/01-neural-network-forward-pass.md) |
| Gradient descent | $\theta \leftarrow \theta - \eta\nabla_\theta L$ | [Gradient descent](../lessons/11-gradient-descent-backprop/01-gradient-descent.md) |
| Sigmoid + BCE error signal | $\partial L/\partial z = \hat{y} - y$ | [Backpropagation](../lessons/11-gradient-descent-backprop/02-backpropagation.md) |
| Conv output size | $\lfloor (H - k + 2p)/s\rfloor + 1$ | [CNNs](../lessons/13-deep-architectures/01-convolutional-networks.md) |
| Self-attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ | [Self-attention](../lessons/14-transformers/01-self-attention.md) |
| Perplexity | $\exp(\text{mean token loss})$ | [Tokenization](../lessons/15-llms/01-tokenization-and-pretraining.md) |
| LoRA | $W = W_0 + \frac{\alpha}{r}BA$ | [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md) |
| nDCG | $\frac{\sum \text{rel}_i/\log_2(i+1)}{\text{IDCG}}$ | [LLM evaluation](../lessons/17-llm-evaluation/01-llm-evaluation.md) |
| Little's law | $L = \lambda W$ | [Production architecture](../lessons/18-production-ai/01-production-architecture.md) |
