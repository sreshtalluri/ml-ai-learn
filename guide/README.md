# From machine learning to LLMs: the GitHub edition

A visual, math-first course for software engineers moving into ML and AI engineering. This folder is the complete course in plain markdown: lessons, figures, runnable Python, quizzes with hidden answers, model cards, cheat sheets, and a glossary. Nothing to install to read it.

Prefer interactive labs and progress tracking? The same content runs as a website: **[sreshtalluri.github.io/ml-ai-learn](https://sreshtalluri.github.io/ml-ai-learn/)**.

## How to use this course

- **Pass 1, intuition.** Read the mental model and study each figure before the formulas.
- **Pass 2, mechanics.** Redo the worked examples by hand, then run the scripts in [`code/`](code/).
- **Pass 3, engineering.** Build the project for each phase and explain the trade-offs out loud.
- **Active recall.** Close the lesson and answer its quiz before opening the answers.

Every lesson has the same six layers: **intuition, visualization, math, implementation, engineering, knowledge check.**

Suggested pace: 12 weeks at 5 to 7 hours per week ([12-week plan](lessons/20-projects/01-twelve-week-plan.md)). Short on time? See the [learning paths](learning-paths.md): a 4-week AI-engineering fast track and an interview-prep pass.

### Run the code

```bash
cd guide
uv run code/03-regression/linear_regression.py   # installs numpy, matplotlib, scikit-learn on first run
```

No uv? `pip install numpy matplotlib scikit-learn` and use `python` instead.

## Curriculum

### Foundations

| Module | Lessons |
|---|---|
| [0. Math and programming foundations](lessons/00-foundations/README.md) | [Vectors and matrices](lessons/00-foundations/01-vectors-and-matrices.md) · [Calculus for ML](lessons/00-foundations/02-calculus-for-ml.md) · [Probability and statistics](lessons/00-foundations/03-probability-and-statistics.md) · [Python toolkit](lessons/00-foundations/04-python-toolkit.md) |
| [1. ML vocabulary and learning paradigms](lessons/01-ml-vocabulary/README.md) | [The language of ML](lessons/01-ml-vocabulary/01-ml-vocabulary.md) · [Learning paradigms](lessons/01-ml-vocabulary/02-learning-paradigms.md) |
| [2. The end-to-end ML workflow](lessons/02-ml-workflow/README.md) | [Workflow, splits, and leakage](lessons/02-ml-workflow/01-ml-workflow.md) · [Overfitting and bias-variance](lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) |

### Classical machine learning

| Module | Lessons |
|---|---|
| [3. Regression](lessons/03-regression/README.md) | [Linear regression](lessons/03-regression/01-linear-regression.md) · [Regularization and regression metrics](lessons/03-regression/02-regularization-and-regression-metrics.md) |
| [4. Classification](lessons/04-classification/README.md) | [Logistic regression](lessons/04-classification/01-logistic-regression.md) · [Classification metrics](lessons/04-classification/02-classification-metrics.md) |
| [5. Instance-based and probabilistic models](lessons/05-instance-and-probabilistic/README.md) | [K-nearest neighbors](lessons/05-instance-and-probabilistic/01-k-nearest-neighbors.md) · [Naive Bayes](lessons/05-instance-and-probabilistic/02-naive-bayes.md) · [Support vector machines](lessons/05-instance-and-probabilistic/03-support-vector-machines.md) |
| [6. Trees and ensembles](lessons/06-trees-and-ensembles/README.md) | [Decision trees](lessons/06-trees-and-ensembles/01-decision-trees.md) · [Random forests and boosting](lessons/06-trees-and-ensembles/02-random-forests-and-boosting.md) |
| [7. Unsupervised learning](lessons/07-unsupervised/README.md) | [K-means](lessons/07-unsupervised/01-k-means.md) · [DBSCAN, hierarchical, and mixtures](lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md) |
| [8. Dimensionality reduction](lessons/08-dimensionality-reduction/README.md) | [PCA, t-SNE, and UMAP](lessons/08-dimensionality-reduction/01-pca.md) |

### NLP and deep learning

| Module | Lessons |
|---|---|
| [9. Classical NLP](lessons/09-classical-nlp/README.md) | [From text to vectors](lessons/09-classical-nlp/01-text-to-vectors.md) · [Word embeddings](lessons/09-classical-nlp/02-word-embeddings.md) |
| [10. Neural-network foundations](lessons/10-neural-networks/README.md) | [The forward pass](lessons/10-neural-networks/01-neural-network-forward-pass.md) |
| [11. Gradient descent and backpropagation](lessons/11-gradient-descent-backprop/README.md) | [Gradient descent](lessons/11-gradient-descent-backprop/01-gradient-descent.md) · [Backpropagation](lessons/11-gradient-descent-backprop/02-backpropagation.md) |
| [12. Training and regularization](lessons/12-training-regularization/README.md) | [Training and regularization](lessons/12-training-regularization/01-training-and-regularization.md) |
| [13. Deep-learning architectures](lessons/13-deep-architectures/README.md) | [CNNs](lessons/13-deep-architectures/01-convolutional-networks.md) · [RNNs, LSTMs, GRUs](lessons/13-deep-architectures/02-recurrent-networks.md) · [Autoencoders, diffusion, transfer learning](lessons/13-deep-architectures/03-autoencoders-diffusion-and-transfer.md) |

### Transformers and LLMs

| Module | Lessons |
|---|---|
| [14. Transformers](lessons/14-transformers/README.md) | [Self-attention](lessons/14-transformers/01-self-attention.md) · [The transformer architecture](lessons/14-transformers/02-transformer-architecture.md) |
| [15. Large language models](lessons/15-llms/README.md) | [Tokenization and pretraining](lessons/15-llms/01-tokenization-and-pretraining.md) · [Decoding](lessons/15-llms/02-decoding.md) · [Adapting LLMs](lessons/15-llms/03-adapting-llms.md) |
| [16. Retrieval-augmented generation](lessons/16-rag/README.md) | [The RAG pipeline](lessons/16-rag/01-rag-pipeline.md) |
| [17. LLM evaluation](lessons/17-llm-evaluation/README.md) | [Evaluating LLM systems](lessons/17-llm-evaluation/01-llm-evaluation.md) |

### AI engineering

| Module | Lessons |
|---|---|
| [18. Production AI engineering](lessons/18-production-ai/README.md) | [Production architecture](lessons/18-production-ai/01-production-architecture.md) · [Reliability, cost, and observability](lessons/18-production-ai/02-reliability-cost-and-observability.md) |
| [19. Safety and security](lessons/19-safety-security/README.md) | [Securing AI systems](lessons/19-safety-security/01-ai-security.md) |
| [20. Projects and career roadmap](lessons/20-projects/README.md) | [12-week plan](lessons/20-projects/01-twelve-week-plan.md) · [Portfolio project ladder](lessons/20-projects/02-project-ladder.md) |

## Reference

- [Model cards](models/README.md): 28 models, each with objective, assumptions, hyperparameters, failure modes, and neighbors.
- [Cheat sheets](cheatsheets/README.md): metrics, losses, optimizers, transformers, RAG, security, formulas, tensor shapes.
- [Glossary](glossary.md): plain-English and formal definitions.
- [Quizzes](quizzes/): one per lesson, answers hidden in collapsible sections.

## The formulas worth remembering

| Concept | Formula |
|---|---|
| Linear model | $\hat{y} = w \cdot x + b$ |
| Mean squared error | $\frac{1}{n}\sum (y - \hat{y})^2$ |
| Sigmoid | $1 / (1 + e^{-z})$ |
| Gradient update | $\theta \leftarrow \theta - \eta \nabla_\theta L$ |
| Cosine similarity | $(a \cdot b) / (\lVert a \rVert \lVert b \rVert)$ |
| TF-IDF | $\text{TF}(t,d) \times \log(N / \text{DF}(t))$ |
| Self-attention | $\text{softmax}(QK^\top / \sqrt{d_k})\,V$ |
| Precision, recall | $TP/(TP+FP)$, $TP/(TP+FN)$ |
| F1 | $2PR / (P + R)$ |

## Contributing

Lessons follow a fixed format so both the GitHub edition and the website can render them. See [AUTHORING.md](AUTHORING.md).
