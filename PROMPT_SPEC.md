You are a senior full-stack engineer, machine learning educator, curriculum designer, data visualization specialist, and UI/UX designer.

I want you to build a complete, locally runnable, interactive learning platform that teaches me machine learning, deep learning, natural language processing, transformers, large language models, and production AI engineering.

This must not be a generic collection of text pages. It should feel like a polished, interactive technical course and visual laboratory that I can use as my personal one-stop learning environment.

The platform should help me:

1. Understand the overall structure of machine learning and AI.
2. See how individual concepts connect to one another.
3. Build strong intuition before introducing formulas.
4. Work through the mathematics step by step.
5. Manipulate interactive visualizations.
6. inspect model behavior and failure modes.
7. Run or study practical Python examples.
8. Test my understanding with exercises and quizzes.
9. Track progress through a structured curriculum.
10. Learn enough theory and engineering to move toward an AI engineer or ML engineer role.

Use the learning guide that I provide as the primary curriculum source. Preserve its important concepts, examples, formulas, terminology, learning plan, and engineering guidance, but reorganize the content when necessary to create a better interactive learning experience.

Do not merely convert the document into HTML.

Build an actual educational product.

==================================================
1. TARGET USER AND TEACHING STYLE
==================================================

The learner is a software engineer moving more deeply into machine learning, deep learning, and AI engineering.

The learner is especially responsive to:

- Visual explanations
- Interactive diagrams
- Concrete examples
- Step-by-step mathematics
- Comparisons between related models
- Explanations of why a technique works
- Explanations of when a technique fails
- Connections between theory and production systems
- Implementations that expose what libraries normally hide

Teach every major concept using the following layered structure:

Level 1: Intuition
Explain the concept in plain English with an analogy or mental model.

Level 2: Visualization
Use an interactive graph, diagram, animation, geometric interpretation, or process visualization.

Level 3: Mathematics
Introduce the formal notation, define every symbol, and solve at least one small numerical example step by step.

Level 4: Implementation
Show a minimal implementation, preferably once from scratch and once using an established library.

Level 5: Engineering
Explain real-world use cases, model-selection considerations, computational cost, production concerns, and failure modes.

Level 6: Knowledge check
Provide conceptual questions, numerical exercises, debugging questions, and one small implementation challenge.

Do not introduce notation without defining it.

Do not skip intermediate mathematical steps.

Whenever possible, let the learner adjust inputs and observe how the output changes.

==================================================
2. PRODUCT VISION
==================================================

Create a responsive web application that combines:

- Interactive textbook
- Visual model explorer
- Math walkthrough system
- Python code reference
- Experiment playground
- Quiz application
- Progress tracker
- Model-selection guide
- AI engineering reference
- Project roadmap

The result should feel closer to a combination of:

- A polished online technical course
- An interactive mathematical textbook
- A machine learning visualization laboratory
- A personal study dashboard
- An AI engineering field guide

The platform must work well on desktop and remain usable on tablet and mobile.

==================================================
3. RECOMMENDED TECHNICAL STACK
==================================================

Use a modern, maintainable stack.

Preferred default:

- Next.js with App Router
- TypeScript with strict mode
- React
- Tailwind CSS
- shadcn/ui components
- MDX or structured TypeScript/JSON content files
- Recharts, Plotly.js, D3.js, or a suitable combination
- KaTeX for mathematical notation
- Framer Motion for restrained educational animations
- Zustand or React Context for lightweight local state
- IndexedDB or localStorage for progress persistence
- Vitest and React Testing Library
- Playwright for critical end-to-end flows

If an interactive simulation is easier to implement using Canvas or SVG, use the most appropriate option.

Avoid adding a backend unless one is genuinely needed.

The initial version should:

- Run locally
- Require no paid APIs
- Require no authentication
- Store progress locally
- Use deterministic example datasets
- Work without an external model API

Architect the platform so cloud sync, authentication, notebooks, or an AI tutor could be added later.

Use Python only if a lightweight backend is truly needed for model execution. If client-side TypeScript can perform a small simulation clearly and reliably, prefer client-side execution.

==================================================
4. INFORMATION ARCHITECTURE
==================================================

Create these primary navigation areas:

A. Home
B. Learning Paths
C. Course Modules
D. Model Explorer
E. Math Lab
F. NLP Lab
G. Neural Network Lab
H. Transformer Lab
I. LLM and RAG Lab
J. AI Engineering
K. Projects
L. Quizzes
M. Cheat Sheets
N. Progress Dashboard
O. Glossary and Formula Reference

The application should provide two navigation modes:

1. Guided learning mode
   A recommended sequence from foundations to production AI.

2. Reference mode
   Direct access to any model, formula, visualization, or engineering concept.

==================================================
5. HOME PAGE
==================================================

Build a strong home page containing:

- A clear title and product explanation
- A visual map from data to models to deployment
- Continue-learning card
- Current module and progress
- Recommended next lesson
- Quick links to interactive labs
- Recently viewed topics
- Skill-area progress:
  - Mathematics
  - Classical machine learning
  - NLP
  - Deep learning
  - Transformers
  - LLMs
  - AI engineering
- A full learning-path overview
- A “How everything connects” visualization

The “How everything connects” section should visually show:

Data
→ representation
→ model
→ prediction
→ loss
→ optimization
→ evaluation
→ deployment
→ monitoring

It should also show how the following branches fit into that lifecycle:

- Regression
- Classification
- Clustering
- Dimensionality reduction
- NLP
- Neural networks
- Transformers
- LLM applications
- Retrieval-augmented generation
- Agents and tool use

Clicking nodes should open relevant lessons.

==================================================
6. CURRICULUM
==================================================

Organize the course into the following modules.

MODULE 0: Mathematical and programming foundations

Topics:

- Scalars, vectors, matrices, and tensors
- Shapes and dimensions
- Dot products
- Matrix multiplication
- Functions and parameters
- Exponents and logarithms
- Derivatives
- Partial derivatives
- Gradients
- Chain rule
- Probability
- Conditional probability
- Expectation
- Variance
- Common probability distributions
- Bayes’ rule
- Sampling
- Correlation versus causation
- Python, NumPy, pandas, and plotting foundations

Interactive elements:

- Vector addition visualizer
- Dot-product geometry
- Matrix shape checker
- Matrix multiplication animation
- Derivative-as-slope explorer
- Gradient vector field
- Probability distribution explorer
- Conditional probability tree

MODULE 1: Machine learning vocabulary and learning paradigms

Topics:

- Sample and observation
- Feature
- Target and label
- Parameter
- Hyperparameter
- Prediction
- Loss
- Metric
- Training
- Inference
- Generalization
- Supervised learning
- Unsupervised learning
- Self-supervised learning
- Semi-supervised learning
- Reinforcement learning

Create an interactive decision guide that helps the learner decide which learning paradigm fits a problem.

Include examples such as:

- House-price prediction
- Fraud detection
- Customer clustering
- Anomaly detection
- Next-token prediction
- Recommendation
- Robot control

MODULE 2: End-to-end machine learning workflow

Topics:

- Problem formulation
- Prediction unit
- Data collection
- Exploratory data analysis
- Missing values
- Duplicates
- Outliers
- Train, validation, and test sets
- Random split
- Time split
- Group split
- Stratification
- Data leakage
- Baselines
- Feature preprocessing
- Reproducibility
- Evaluation
- Error analysis
- Deployment
- Monitoring

Interactive elements:

- Dataset splitting visualizer
- Leakage detective game
- Train versus validation curve explorer
- Underfitting versus overfitting visualization
- Bias-variance visualization
- Experiment-design checklist

MODULE 3: Regression

Topics:

- Linear regression
- Multiple linear regression
- Prediction equations
- Residuals
- Mean squared error
- Mean absolute error
- Root mean squared error
- R-squared
- Mean absolute percentage error
- Polynomial features
- Ridge regression
- Lasso regression
- Elastic Net
- Assumptions of linear regression
- Feature scaling
- Multicollinearity
- Outliers

Required interactive lab:

- Scatterplot with draggable observations
- Adjustable slope and bias
- Live prediction line
- Visible residual lines
- Live MSE and MAE calculations
- Best-fit solution
- Toggle between L1 and L2 penalties
- Visual demonstration of how regularization changes weights

Required worked example:

Given a small dataset, calculate:

1. Predictions
2. Residuals
3. Squared errors
4. MSE
5. Gradient with respect to slope
6. Gradient with respect to bias
7. One gradient-descent update

Show every arithmetic step.

MODULE 4: Classification

Topics:

- Binary classification
- Multiclass classification
- Logistic regression
- Logits
- Sigmoid
- Softmax
- Binary cross-entropy
- Categorical cross-entropy
- Decision boundaries
- Classification thresholds
- Confusion matrix
- Accuracy
- Precision
- Recall
- Specificity
- F1 score
- ROC curve
- Precision-recall curve
- Calibration
- Class imbalance

Required interactive lab:

- Adjustable classification threshold
- Live confusion matrix
- Live precision, recall, accuracy, and F1
- Visualization of false positives and false negatives
- Business-cost controls for different errors
- Probability calibration visualization

MODULE 5: Instance-based and probabilistic models

Topics:

- K-nearest neighbors
- Distance metrics
- Euclidean distance
- Manhattan distance
- Choosing K
- Feature scaling
- Curse of dimensionality
- Naive Bayes
- Bayes’ rule
- Prior probability
- Likelihood
- Posterior probability
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Support vector machines
- Margins
- Kernels
- Linear versus nonlinear decision boundaries

Required visualizations:

- KNN neighborhood explorer
- Adjustable K
- Distance comparison
- Feature-scaling effect
- SVM margin visualization
- Kernel transformation intuition
- Naive Bayes probability calculator

MODULE 6: Trees and ensembles

Topics:

- Decision trees
- Splitting
- Gini impurity
- Entropy
- Information gain
- Tree depth
- Pruning
- Random forests
- Bootstrap sampling
- Random feature selection
- Bagging
- Boosting
- Gradient boosting
- XGBoost
- LightGBM
- CatBoost
- Learning rate
- Number of estimators
- Maximum depth
- Subsampling
- Regularization
- Feature importance
- SHAP overview

Required interactive lab:

- Build a small decision tree one split at a time
- Calculate Gini impurity before and after a split
- Compare a single tree with a random forest
- Animate residual correction in gradient boosting
- Show how learning rate and estimator count interact
- Display underfitting and overfitting for different tree depths

Include a worked gradient-boosting example using a tiny regression dataset.

MODULE 7: Unsupervised learning

Topics:

- K-means clustering
- Centroids
- Assignment and update steps
- K-means objective
- Initialization
- Elbow method
- Silhouette score
- Hierarchical clustering
- Dendrograms
- DBSCAN
- Density
- Epsilon
- Minimum samples
- Gaussian mixture models
- Soft cluster membership
- Anomaly detection

Required interactive labs:

- Place points and centroids on a canvas
- Animate K-means assignment and centroid updates
- Step through iterations manually
- Change K and initialization
- Compare K-means and DBSCAN
- Adjust DBSCAN epsilon and minimum samples
- View Gaussian mixture soft probabilities
- Identify when clustering assumptions fail

MODULE 8: Dimensionality reduction

Topics:

- High-dimensional data
- Variance
- Covariance
- Principal component analysis
- Eigenvectors and eigenvalues at an intuitive level
- Projection
- Explained variance
- t-SNE
- UMAP
- Visualization caveats
- Compression and denoising

Required visualization:

- Rotatable 2D point cloud
- Principal component direction
- Projection onto the component
- Explained variance
- Before and after dimensionality reduction
- Warning explaining why visual clusters do not automatically prove true categories

MODULE 9: Classical NLP

Topics:

- Text preprocessing
- Unicode normalization
- Tokenization
- Stop words
- Stemming
- Lemmatization
- Bag-of-words
- Vocabulary
- Document-term matrices
- N-grams
- TF-IDF
- Cosine similarity
- Word2Vec
- Skip-gram
- CBOW
- Static embeddings
- Contextual embeddings

Required interactive labs:

- Live tokenizer
- Bag-of-words matrix builder
- N-gram explorer
- TF-IDF calculator
- Cosine-similarity visualizer
- Small semantic vector space
- Compare TF-IDF with embeddings

For TF-IDF, allow the learner to edit a corpus and show:

1. Term frequency
2. Document frequency
3. Inverse document frequency
4. Final TF-IDF
5. A worked calculation for one selected word

MODULE 10: Neural-network foundations

Topics:

- Perceptron
- Neuron
- Weights
- Biases
- Layers
- Forward propagation
- Linear transformations
- Activations
- ReLU
- Sigmoid
- Tanh
- GELU
- Softmax
- Loss functions
- Parameters
- Initialization
- Batches
- Epochs

Required neural-network visualizer:

- Adjustable number of input, hidden, and output neurons
- Animated forward pass
- Visible weights and biases
- Node activations
- Tensor shapes
- Numeric calculations for a small network
- Toggle activation functions
- See how changing a weight affects the output

MODULE 11: Gradient descent and backpropagation

Topics:

- Objective functions
- Gradients
- Learning rate
- Batch gradient descent
- Stochastic gradient descent
- Mini-batch gradient descent
- Momentum
- Adam
- Computational graphs
- Chain rule
- Backpropagation
- Automatic differentiation
- Vanishing gradients
- Exploding gradients
- Gradient clipping

Required interactive lab:

- Loss-surface visualization
- Parameter position
- Gradient arrow
- Adjustable learning rate
- Step-by-step updates
- Comparison of SGD, momentum, and Adam
- Divergence demonstration for excessive learning rate

Required worked backpropagation example:

Use a one-neuron binary classifier with:

x1 = 2
x2 = 1
w1 = 0.5
w2 = -1
b = 0
y = 1

Show:

1. Weighted sum z
2. Sigmoid output
3. Binary cross-entropy
4. dL/dz
5. dL/dw1
6. dL/dw2
7. dL/db
8. Parameter update
9. New prediction

Add an option to change the values and recompute every step.

MODULE 12: Training and regularization

Topics:

- Training loss
- Validation loss
- Generalization
- Weight decay
- L1 and L2 regularization
- Dropout
- Early stopping
- Batch normalization
- Layer normalization
- Data augmentation
- Learning-rate schedules
- Hyperparameter tuning
- Grid search
- Random search
- Bayesian optimization
- Cross-validation

Required visualizations:

- Training and validation curves
- Early stopping point
- Dropout animation
- Regularization effect
- Learning-rate schedule plots
- Hyperparameter search comparison

MODULE 13: Deep-learning architectures

Topics:

- Multilayer perceptrons
- Convolutional neural networks
- Kernels and filters
- Stride
- Padding
- Pooling
- Feature maps
- Recurrent neural networks
- Hidden state
- LSTM
- GRU
- Sequence modeling
- Autoencoders
- Latent spaces
- Diffusion models
- Transfer learning

Required visualizations:

- Convolution kernel moving over an image matrix
- Resulting feature map calculation
- Pooling calculation
- RNN unrolled through time
- LSTM gate overview
- Autoencoder bottleneck
- Diffusion forward-noising and reverse-denoising process

MODULE 14: Transformers

Topics:

- Limitations of recurrent models
- Attention
- Self-attention
- Query
- Key
- Value
- Dot-product similarity
- Scaling by square root of key dimension
- Softmax attention weights
- Weighted value sum
- Multi-head attention
- Positional encoding
- Residual connections
- Layer normalization
- Feed-forward networks
- Encoder blocks
- Decoder blocks
- Cross-attention
- Causal masking
- Encoder-only models
- Decoder-only models
- Encoder-decoder models

Required transformer lab:

- Enter a short sequence
- Show tokens and embeddings
- Display query, key, and value matrices
- Calculate QK transpose
- Apply scaling
- Apply causal mask
- Apply softmax
- Produce weighted value outputs
- Display attention as a heatmap
- Move through every operation step by step
- Show tensor shapes at every stage

Required numerical example:

q = [1, 0]
k1 = [1, 0]
k2 = [0, 1]
v1 = [10, 0]
v2 = [0, 6]
dk = 2

Calculate the complete attention output while showing every step.

Include a transformer-block architecture explorer that visually demonstrates residual paths, normalization, attention, and the feed-forward network.

MODULE 15: Large language models

Topics:

- Tokens
- Token IDs
- Embedding tables
- Context windows
- Autoregressive pretraining
- Next-token prediction
- Cross-entropy
- Logits
- Softmax
- Perplexity
- Checkpoints
- Inference
- Greedy decoding
- Temperature
- Top-k
- Top-p
- Beam search
- Pretraining
- Instruction tuning
- Fine-tuning
- LoRA
- PEFT
- Preference optimization
- Alignment
- Hallucinations
- Calibration and uncertainty

Required interactive labs:

- Tokenization explorer
- Next-token probability distribution
- Logit-to-softmax calculator
- Temperature adjustment
- Top-k and top-p filtering
- Context-window usage meter
- Autoregressive generation animation
- Comparison of prompting, RAG, LoRA, and full fine-tuning

Teach clearly that an LLM produces probability distributions over tokens and is not inherently a factual database.

MODULE 16: Retrieval-augmented generation

Topics:

- Document ingestion
- Parsing
- Cleaning
- Chunking
- Chunk size
- Chunk overlap
- Embeddings
- Vector indexes
- Dense retrieval
- Sparse retrieval
- BM25
- Hybrid search
- Metadata filtering
- Reranking
- Context construction
- Grounded generation
- Citations
- Retrieval evaluation
- Answer evaluation

Required RAG lab:

- Provide a small built-in document collection
- Split documents into chunks
- Change chunk size and overlap
- Display chunk embeddings conceptually
- Enter a query
- Show retrieved chunks
- Display similarity scores
- Rerank results
- Construct final context
- Show which answer claims are supported
- Demonstrate failure when retrieval misses the correct chunk

Clearly explain:

- RAG changes runtime context
- Fine-tuning changes model weights
- RAG is often preferable for private or frequently changing facts
- Fine-tuning is useful for changing behavior, format, or specialization

MODULE 17: LLM evaluation

Topics:

- Task accuracy
- Exact match
- F1
- Rubric-based evaluation
- Retrieval recall at K
- Mean reciprocal rank
- nDCG
- Context relevance
- Groundedness
- Faithfulness
- Citation correctness
- Hallucination rate
- Human evaluation
- LLM-as-judge limitations
- Latency
- Throughput
- Tokens per second
- Cost per request
- Cache hit rate
- Safety evaluation
- Prompt-injection testing

Required evaluation dashboard:

- Multiple test cases
- Expected outputs
- Retrieved contexts
- Generated answer field
- Per-case scores
- Aggregate metrics
- Failure categorization
- Latency and cost fields
- Side-by-side experiment comparison

MODULE 18: Production AI engineering

Topics:

- APIs
- Batch versus online inference
- Authentication
- Authorization
- Rate limits
- Model routing
- Prompt templates
- Structured outputs
- Schemas
- Validation
- Tool calling
- Retries
- Timeouts
- Circuit breakers
- Idempotency
- Caching
- Observability
- Tracing
- Data drift
- Feature drift
- Prompt drift
- Model versioning
- Index versioning
- Training-serving skew
- Latency
- Throughput
- Scalability
- Cost optimization
- Quantization
- Distillation
- Batching
- KV caching

Create an interactive production architecture diagram containing:

- Client
- API gateway
- Authentication
- Orchestration
- Retrieval
- Model
- Tools
- Evaluation
- Cache
- Logging
- Monitoring
- Governance

Clicking a component should explain:

- Responsibility
- Inputs and outputs
- Common failures
- Metrics
- Security considerations
- Scaling considerations

MODULE 19: Safety and security

Topics:

- Prompt injection
- Indirect prompt injection
- Data exfiltration
- Tool abuse
- Excessive permissions
- Secret leakage
- Unsafe generated SQL or code
- Input validation
- Output validation
- Sandboxing
- Allowlists
- Least privilege
- Privacy
- Audit logging
- Human review
- Threat modeling
- Supply-chain risks

Required security lab:

- Show a benign retrieved document
- Show a document containing malicious instructions
- Demonstrate how trusted instructions and untrusted content should remain separated
- Explain how validation, permissions, and allowlists reduce risk
- Never execute unsafe actions in the demonstration

MODULE 20: Projects and career roadmap

Include a progressive project ladder:

1. Tabular model failure-analysis lab
2. Semantic incident-triage system
3. RAG evaluation workbench
4. Adaptive model router
5. Agent reliability sandbox

For every project include:

- Problem statement
- Learning goals
- Architecture
- Dataset options
- Milestones
- Minimum viable version
- Stretch goals
- Evaluation plan
- Testing plan
- Deployment plan
- Security considerations
- Resume bullet examples
- Interview talking points

==================================================
7. MODEL EXPLORER
==================================================

Create a searchable model explorer containing model cards for:

- Linear regression
- Ridge regression
- Lasso regression
- Logistic regression
- K-nearest neighbors
- Naive Bayes
- Support vector machine
- Decision tree
- Random forest
- Gradient boosting
- XGBoost
- K-means
- DBSCAN
- Hierarchical clustering
- Gaussian mixture model
- PCA
- Multilayer perceptron
- CNN
- RNN
- LSTM
- GRU
- Autoencoder
- Transformer encoder
- Transformer decoder
- Large language model
- Embedding model
- Reranker
- Diffusion model

Each model card must include:

- Problem type
- Input
- Output
- Mental model
- Core equation or objective
- Training process
- Required preprocessing
- Important assumptions
- Key hyperparameters
- Good use cases
- Poor use cases
- Strengths
- Weaknesses
- Computational considerations
- Evaluation metrics
- Common failure modes
- Minimal implementation
- Comparison with neighboring models
- Links to relevant lessons and labs

Add filters for:

- Supervised
- Unsupervised
- Self-supervised
- Regression
- Classification
- Clustering
- Dimensionality reduction
- NLP
- Vision
- Generative
- Interpretable
- Low-latency
- Small-data friendly

==================================================
8. MATH LAB
==================================================

Create a dedicated Math Lab.

Include interactive calculators for:

- Dot product
- Matrix multiplication
- Euclidean distance
- Mean and variance
- Standardization
- Sigmoid
- Softmax
- MSE
- MAE
- Binary cross-entropy
- Categorical cross-entropy
- Gradient update
- Cosine similarity
- TF-IDF
- Gini impurity
- Entropy
- Precision
- Recall
- F1
- Attention

Every calculator should:

- Let the learner edit the input
- Show the formula
- Define every symbol
- Substitute actual numbers
- Show intermediate arithmetic
- Show the final result
- Explain what the result means
- Warn about common mistakes
- Link to the related lesson

==================================================
9. VISUALIZATION STANDARDS
==================================================

Visualizations must be educational, not decorative.

Every visualization should have:

- A clear title
- Axis labels when relevant
- Legend when relevant
- Controls with labels
- Reset button
- Example presets
- Explanation of what changed
- A short interpretation
- Accessible colors
- Dark-mode support
- Keyboard accessibility where practical

Prefer SVG or Canvas for interactive mathematical diagrams.

Use smooth but restrained animations. Animations should explain changes rather than merely decorate the interface.

Do not use fabricated benchmark results.

Use deterministic synthetic datasets and label them clearly as synthetic.

==================================================
10. LESSON PAGE DESIGN
==================================================

Each lesson page should contain:

1. Learning objectives
2. Prerequisites
3. Why the topic matters
4. Mental model
5. Interactive visualization
6. Formal explanation
7. Formula breakdown
8. Worked numerical example
9. From-scratch implementation
10. Library implementation
11. Common mistakes
12. Failure modes
13. Production considerations
14. Knowledge checks
15. Practice exercise
16. Summary
17. Links to related lessons
18. Suggested next lesson

Add a sticky sidebar or local table of contents for long lessons.

Add “Intuition,” “Math,” “Code,” and “Engineering” tabs when that improves usability.

==================================================
11. QUIZZES AND ACTIVE RECALL
==================================================

Include:

- Multiple-choice questions
- Multi-select questions
- Match-the-concept questions
- Calculation exercises
- Fill-in-the-equation exercises
- Arrange-the-process questions
- Debugging questions
- Short-answer reflection questions

Quiz features:

- Immediate explanations
- Why the correct answer is correct
- Why alternatives are incorrect
- Difficulty labels
- Retry incorrect questions
- Topic mastery score
- Review queue
- Local progress persistence

Do not rely only on recognition questions. Include questions that require derivation, comparison, diagnosis, and design decisions.

==================================================
12. PROGRESS SYSTEM
==================================================

Track:

- Lessons completed
- Labs completed
- Quiz scores
- Topics needing review
- Modules completed
- Time spent, if practical
- Current streak, if practical
- Project milestones
- Mastery by skill area

Allow:

- Mark as complete
- Bookmark page
- Add personal note
- Reset progress
- Export progress as JSON
- Import progress from JSON

Use local storage or IndexedDB for the first version.

==================================================
13. CHEAT SHEETS
==================================================

Create concise reference pages for:

- Model selection
- Classification metrics
- Regression metrics
- Loss functions
- Activation functions
- Optimizers
- Regularization
- Clustering algorithms
- NLP representations
- Transformer architecture
- LLM adaptation
- RAG pipeline
- LLM evaluation
- Production reliability
- Security controls
- Important formulas
- Tensor shapes
- Common failure modes

Each cheat sheet should link back to complete lessons.

==================================================
14. GLOSSARY
==================================================

Create a searchable glossary covering important terms such as:

- Parameter
- Hyperparameter
- Feature
- Label
- Loss
- Metric
- Epoch
- Batch
- Gradient
- Logit
- Activation
- Embedding
- Token
- Context window
- Attention
- Checkpoint
- Fine-tuning
- RAG
- Grounding
- Inference
- Quantization
- Distillation
- Tool calling
- Agent
- Drift
- Calibration

Every glossary entry should include:

- Plain-English definition
- Formal definition where useful
- Small example
- Related terms
- Link to lesson

==================================================
15. CONTENT AND PEDAGOGICAL RULES
==================================================

Follow these rules throughout the application:

- Start with intuition before equations.
- Define every mathematical symbol.
- Explain tensor shapes.
- Work through arithmetic explicitly.
- Never say “the math works out” without showing how.
- Compare complex models against simple baselines.
- Explain assumptions.
- Explain failure modes.
- Explain preprocessing requirements.
- Distinguish loss functions from evaluation metrics.
- Distinguish parameters from hyperparameters.
- Distinguish training from inference.
- Distinguish RAG from fine-tuning.
- Distinguish prediction confidence from factual certainty.
- Distinguish correlation from causation.
- Distinguish a discovered cluster from a verified real-world category.
- Label synthetic examples.
- Mention computational complexity when relevant.
- Include production consequences where relevant.
- Use concise but technically accurate language.
- Avoid unnecessary jargon.
- Do not oversimplify to the point of being incorrect.

==================================================
16. UI AND DESIGN
==================================================

Use a polished educational visual language.

Suggested characteristics:

- Professional and calm
- Strong typography
- Generous spacing
- Clear information hierarchy
- Restrained blue, teal, purple, and orange accents
- Light and dark themes
- Cards for model summaries
- Callouts for intuition, warnings, and engineering notes
- Code blocks with copy buttons
- Equation blocks with copyable LaTeX
- Responsive diagrams
- Consistent chart colors
- Visible focus states
- Accessible contrast

Do not make the interface look like a generic admin dashboard.

Do not overload pages with cards.

Use progressive disclosure for advanced details.

==================================================
17. DATA AND CONTENT ARCHITECTURE
==================================================

Keep educational content separated from presentation logic.

Use a structured lesson schema containing fields such as:

- id
- slug
- module
- title
- summary
- prerequisites
- objectives
- mentalModel
- sections
- formulas
- examples
- visualizations
- codeExamples
- exercises
- quizQuestions
- commonMistakes
- productionNotes
- relatedLessons
- references

Create reusable components for:

- Lesson header
- Mental-model callout
- Formula walkthrough
- Worked example
- Interactive visualization
- Code comparison
- Common-mistake warning
- Model card
- Quiz
- Progress control
- Glossary tooltip
- Related-concept links

Do not hardcode every lesson into one large page component.

==================================================
18. QUALITY REQUIREMENTS
==================================================

The application must:

- Run without TypeScript errors
- Pass linting
- Avoid broken routes
- Avoid console errors
- Include useful empty states
- Include loading states where needed
- Be reasonably accessible
- Work in light and dark mode
- Be responsive
- Persist progress
- Include tests for core calculations
- Include tests for progress persistence
- Include tests for at least one interactive lab
- Include a README with setup and architecture instructions

Validate mathematical functions with unit tests.

Important functions to test include:

- Sigmoid
- Softmax
- MSE
- Binary cross-entropy
- TF-IDF
- Cosine similarity
- Gini impurity
- Precision
- Recall
- F1
- One gradient-descent update
- Scaled dot-product attention

==================================================
19. DELIVERY STRATEGY
==================================================

Do not attempt to generate the entire platform blindly in one uncontrolled step.

First:

1. Inspect the existing repository.
2. Summarize the current structure.
3. Propose a scalable architecture.
4. Create an implementation plan.
5. Identify reusable components.
6. Define content schemas.
7. Define routes.
8. Define milestones.

Then implement in phases.

Phase 1: Foundation

- Project setup
- Design system
- Navigation
- Theme
- Content schema
- Homepage
- Learning-path page
- Lesson template
- Progress persistence
- Glossary tooltips

Phase 2: Classical machine learning

- Foundations
- Supervised versus unsupervised learning
- Workflow
- Regression
- Classification
- KNN
- Trees
- Ensembles
- Clustering
- PCA

Phase 3: NLP and deep learning

- Bag-of-words
- TF-IDF
- Embeddings
- Neural networks
- Losses
- Gradient descent
- Backpropagation
- Regularization
- CNNs and RNNs

Phase 4: Transformers and LLMs

- Attention lab
- Transformer blocks
- Tokenization
- Decoding
- Pretraining
- Fine-tuning
- LoRA
- RAG
- Evaluation

Phase 5: AI engineering

- Production architecture
- Observability
- Cost and latency
- Reliability
- Security
- Projects
- Model-selection guide

Phase 6: Quality and polish

- Tests
- Accessibility
- Responsive behavior
- Content review
- Performance
- Documentation

After each phase:

- Run type checking
- Run linting
- Run tests
- Start the application
- Check the browser console
- Verify major routes
- Summarize completed work
- List known limitations
- Recommend the next phase

==================================================
20. FIRST IMPLEMENTATION MILESTONE
==================================================

For the first working milestone, implement:

1. Complete responsive application shell
2. Home dashboard
3. Course navigation
4. Structured content system
5. Progress persistence
6. Model explorer shell
7. Math Lab shell
8. Glossary
9. Lesson template
10. Full initial lessons for:
   - Supervised versus unsupervised learning
   - Linear regression
   - K-nearest neighbors
   - K-means clustering
   - Gradient descent
   - Neural-network forward pass
   - Self-attention
11. Interactive labs for:
   - Linear regression
   - KNN
   - K-means
   - Gradient descent
   - Self-attention
12. Quiz system
13. Model-selection cheat sheet
14. Unit tests for all calculations used by those labs
15. README

Do not leave placeholder text such as “content goes here.”

If the full curriculum cannot be completed in one session, complete the first milestone at production quality and create a clear backlog for remaining modules.

==================================================
21. WORKING BEHAVIOR
==================================================

When making implementation choices:

- Favor correctness over visual novelty.
- Favor maintainability over shortcuts.
- Favor reusable teaching components over one-off pages.
- Favor interactive intuition over decorative animation.
- Favor transparent calculations over black-box model output.
- Favor complete, polished vertical slices over many incomplete pages.

When something is ambiguous, make a reasonable engineering decision and document it rather than repeatedly stopping for confirmation.

Before editing files, provide:

1. Repository assessment
2. Proposed architecture
3. Route map
4. Content schema
5. Component plan
6. Implementation phases

Then begin implementation.

At the end of every implementation session, provide:

- What was built
- Files added or changed
- Commands run
- Test results
- How to launch the site
- Known limitations
- Recommended next steps

Begin by inspecting the repository and creating the development plan.
