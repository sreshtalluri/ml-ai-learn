// Lab metadata (server-safe). Components live in src/components/labs/registry-*.tsx.
export type LabArea = "foundations" | "classical-ml" | "nlp" | "neural-networks" | "transformers" | "llm-rag" | "engineering";

export const LAB_AREAS: Record<LabArea, { title: string; blurb: string }> = {
  foundations: { title: "Foundations", blurb: "Choosing a learning paradigm and building valid experiments." },
  "classical-ml": { title: "Classical ML", blurb: "Regression, classification, neighbors, trees, clustering, and PCA." },
  nlp: { title: "NLP Lab", blurb: "Turning text into vectors you can compare." },
  "neural-networks": { title: "Neural Network Lab", blurb: "Forward passes, gradients, backprop, and convolution." },
  transformers: { title: "Transformer Lab", blurb: "Self-attention computed one operation at a time." },
  "llm-rag": { title: "LLM and RAG Lab", blurb: "Decoding, sampling, and retrieval-augmented generation." },
  engineering: { title: "AI Engineering", blurb: "Production architecture and security for AI systems." },
};

export interface LabMeta { id: string; title: string; area: LabArea; lesson: string; description: string }

export const LABS: LabMeta[] = [
  { id: "paradigm-guide", title: "Which learning paradigm?", area: "foundations", lesson: "learning-paradigms", description: "Answer a few questions about a problem and see which paradigm fits, with seven worked scenarios." },
  { id: "fit-explorer", title: "Underfitting vs overfitting", area: "foundations", lesson: "overfitting-and-bias-variance", description: "Change polynomial degree and training size; watch training and validation error diverge." },
  { id: "data-split", title: "Dataset splits and leakage", area: "foundations", lesson: "ml-workflow", description: "Compare random, time, and group splits on a synthetic dataset and spot the leak." },
  { id: "linear-regression", title: "Linear regression", area: "classical-ml", lesson: "linear-regression", description: "Drag points, tilt the line, see residuals, MSE and MAE live, and toggle L1/L2 penalties." },
  { id: "threshold", title: "Classification threshold", area: "classical-ml", lesson: "classification-metrics", description: "Move the threshold and watch the confusion matrix, precision, recall, F1, and business cost change." },
  { id: "knn", title: "K-nearest neighbors", area: "classical-ml", lesson: "k-nearest-neighbors", description: "Move a query point, change K and the distance metric, and see the effect of feature scaling." },
  { id: "decision-tree", title: "Decision tree splits", area: "classical-ml", lesson: "decision-trees", description: "Pick split thresholds and compute Gini impurity before and after, one split at a time." },
  { id: "boosting", title: "Gradient boosting", area: "classical-ml", lesson: "random-forests-and-boosting", description: "Watch each small tree fit the residuals; tune learning rate and number of estimators." },
  { id: "kmeans", title: "K-means clustering", area: "classical-ml", lesson: "k-means", description: "Place centroids, step through assignment and update, change K and initialization." },
  { id: "dbscan", title: "DBSCAN vs K-means", area: "classical-ml", lesson: "density-hierarchical-and-mixture-clustering", description: "Adjust epsilon and min samples on shapes where K-means fails." },
  { id: "pca", title: "PCA projection", area: "classical-ml", lesson: "pca", description: "Rotate a projection axis over a 2D cloud and find the direction of maximum variance." },
  { id: "tfidf", title: "TF-IDF calculator", area: "nlp", lesson: "text-to-vectors", description: "Edit a corpus and see term frequency, document frequency, IDF, and TF-IDF for every word." },
  { id: "embeddings", title: "Semantic vector space", area: "nlp", lesson: "word-embeddings", description: "Explore a small 2D word space and compare cosine similarity with TF-IDF overlap." },
  { id: "nn-forward", title: "Neural network forward pass", area: "neural-networks", lesson: "neural-network-forward-pass", description: "Change layer sizes, weights and activations; follow every number and tensor shape." },
  { id: "gradient-descent", title: "Gradient descent", area: "neural-networks", lesson: "gradient-descent", description: "Step across a loss surface with SGD, momentum, or Adam; push the learning rate until it diverges." },
  { id: "backprop", title: "Backpropagation, step by step", area: "neural-networks", lesson: "backpropagation", description: "A one-neuron classifier: edit inputs and weights and recompute all nine steps." },
  { id: "convolution", title: "Convolution and pooling", area: "neural-networks", lesson: "convolutional-networks", description: "Slide a kernel over a small image, compute each feature-map cell, then max-pool." },
  { id: "attention", title: "Self-attention", area: "transformers", lesson: "self-attention", description: "Enter a sequence and step through Q, K, V, QKᵀ, scaling, masking, softmax, and the output." },
  { id: "decoding", title: "Temperature, top-k, top-p", area: "llm-rag", lesson: "decoding", description: "Turn logits into a next-token distribution and see how each sampling knob reshapes it." },
  { id: "rag", title: "RAG pipeline", area: "llm-rag", lesson: "rag-pipeline", description: "Chunk documents, retrieve for a query, rerank, build context, and see a retrieval miss." },
  { id: "architecture", title: "Production AI architecture", area: "engineering", lesson: "production-architecture", description: "Click each component of an LLM system to see responsibilities, failures, metrics, and security." },
  { id: "prompt-injection", title: "Prompt injection", area: "engineering", lesson: "ai-security", description: "See how untrusted retrieved text tries to hijack instructions, and which controls stop it." },
];

export const getLab = (id: string) => LABS.find((l) => l.id === id);
