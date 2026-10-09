// Lab metadata (server-safe). Components are registered in src/components/labs/registry.tsx.
export type LabArea = "foundations" | "classical-ml" | "nlp" | "neural-networks" | "transformers" | "llm" | "engineering";

export const LAB_AREAS: Record<LabArea, { title: string; blurb: string }> = {
  foundations: { title: "Foundations", blurb: "Choosing a paradigm and seeing over- and underfitting." },
  "classical-ml": { title: "Classical ML", blurb: "Regression, thresholds, neighbors, trees, boosting, clustering, and PCA." },
  nlp: { title: "NLP Lab", blurb: "Turning text into vectors you can compare." },
  "neural-networks": { title: "Neural Network Lab", blurb: "Forward passes, gradient descent, backpropagation, and convolution." },
  transformers: { title: "Transformer Lab", blurb: "Self-attention computed one operation at a time." },
  llm: { title: "LLM and RAG Lab", blurb: "Tokens, decoding, and retrieval-augmented generation." },
  engineering: { title: "AI Engineering", blurb: "Production architecture and security for systems built around models." },
};

export interface LabMeta { id: string; title: string; area: LabArea; lesson: string; description: string }

export const LABS: LabMeta[] = [
  { id: "paradigm-guide", title: "Which learning paradigm?", area: "foundations", lesson: "learning-paradigms", description: "Answer three questions about a problem and see which paradigm fits, with seven worked scenarios." },
  { id: "fit-explorer", title: "Underfitting vs overfitting", area: "foundations", lesson: "overfitting-and-bias-variance", description: "Change polynomial degree, training size, and noise; watch training and validation error split apart." },
  { id: "linear-regression", title: "Linear regression", area: "classical-ml", lesson: "linear-regression", description: "Drag points, tilt the line, see residuals, MSE and MAE live, and toggle L1/L2 penalties." },
  { id: "threshold", title: "Classification threshold", area: "classical-ml", lesson: "classification-metrics", description: "Move the threshold and watch the confusion matrix, precision, recall, F1, and business cost change." },
  { id: "knn", title: "K-nearest neighbors", area: "classical-ml", lesson: "k-nearest-neighbors", description: "Move a query point, change K and the distance metric, and see what feature scaling does." },
  { id: "decision-tree", title: "Build a decision tree", area: "classical-ml", lesson: "decision-trees", description: "Pick features and thresholds, see Gini before and after each split, and grow the tree region by region." },
  { id: "boosting", title: "Gradient boosting, round by round", area: "classical-ml", lesson: "random-forests-and-boosting", description: "Watch each stump fit the residuals; tune learning rate and rounds and find where test error turns up." },
  { id: "kmeans", title: "K-means clustering", area: "classical-ml", lesson: "k-means", description: "Place centroids, step through assignment and update, and change K and the initialization." },
  { id: "pca", title: "Find the first principal component", area: "classical-ml", lesson: "pca", description: "Rotate a projection axis over a 2D cloud and see variance kept versus thrown away." },
  { id: "tfidf", title: "TF-IDF calculator", area: "nlp", lesson: "text-to-vectors", description: "Edit a corpus and see term frequency, document frequency, IDF, and TF-IDF for every word." },
  { id: "nn-forward", title: "Neural network forward pass", area: "neural-networks", lesson: "neural-network-forward-pass", description: "Change weights and activations in a 2-3-1 network and follow every number and tensor shape." },
  { id: "gradient-descent", title: "Gradient descent", area: "neural-networks", lesson: "gradient-descent", description: "Step across a loss surface with SGD, momentum, or Adam, and push the learning rate until it diverges." },
  { id: "backprop", title: "Backpropagation, step by step", area: "neural-networks", lesson: "backpropagation", description: "A one-neuron classifier: edit the inputs and weights and recompute all nine steps." },
  { id: "convolution", title: "Convolution and pooling", area: "neural-networks", lesson: "convolutional-networks", description: "Step a 3×3 kernel over an image, see each multiply-and-add, then max-pool the feature map." },
  { id: "attention", title: "Self-attention", area: "transformers", lesson: "self-attention", description: "Step through Q, K, V, QKᵀ, scaling, the causal mask, softmax, and the weighted sum, with shapes." },
  { id: "decoding", title: "Temperature, top-k, top-p", area: "llm", lesson: "decoding", description: "Turn logits into a next-token distribution and watch each sampling knob reshape it." },
  { id: "tokenizer", title: "Tokenization explorer", area: "llm", lesson: "tokenization-and-pretraining", description: "Type text, step through BPE merges, and watch tokens fill a context window." },
  { id: "rag", title: "RAG pipeline", area: "llm", lesson: "rag-pipeline", description: "Chunk documents, retrieve and rerank for a question, and see what happens when retrieval misses." },
  { id: "architecture", title: "Production LLM architecture", area: "engineering", lesson: "production-architecture", description: "Click components for failures, metrics, and security; trace requests through normal, cache, outage, and tool scenarios." },
  { id: "prompt-injection", title: "Prompt injection", area: "engineering", lesson: "ai-security", description: "Run mocked attacks hidden in retrieved documents and switch on defenses to see which layer stops each one." },
  { id: "lora", title: "LoRA and fine-tuning memory", area: "llm", lesson: "fine-tuning-in-practice", description: "Pick a model size, rank, and precision; compare trainable parameters and GPU memory for full fine-tuning, LoRA, and QLoRA." },
  { id: "vector-search", title: "Approximate nearest neighbors", area: "llm", lesson: "vector-search", description: "Partition vectors into clusters, probe a few, and trade recall against the number of distance computations." },
  { id: "kv-cache", title: "KV cache and inference memory", area: "llm", lesson: "llm-inference", description: "Size the weights and KV cache for a model, context, and batch; see whether it fits and how fast decoding can go." },
  { id: "batching", title: "Static vs continuous batching", area: "llm", lesson: "serving-llms", description: "Simulate requests of different lengths on a GPU and compare throughput and latency under each batching policy." },
  { id: "agent-loop", title: "Agent loop", area: "engineering", lesson: "tool-use-and-agents", description: "Step through an agent's think, call tool, observe loop with mocked tools; inject failures and step limits." },
  { id: "recsys-funnel", title: "Recommendation funnel", area: "engineering", lesson: "ml-system-design", description: "Size candidate generation, ranking, and re-ranking stages against a latency budget and watch recall and cost move." },
  { id: "ab-test", title: "A/B test power and peeking", area: "engineering", lesson: "experimentation-and-ab-testing", description: "Compute sample size from baseline, effect, and power, then simulate how peeking inflates false positives." },
];

export const getLab = (id: string) => LABS.find((l) => l.id === id);
