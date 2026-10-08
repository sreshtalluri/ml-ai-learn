// Lab metadata (server-safe). Components are registered in src/components/labs/registry.tsx.
export type LabArea = "foundations" | "classical-ml" | "nlp" | "neural-networks" | "transformers" | "llm";

export const LAB_AREAS: Record<LabArea, { title: string; blurb: string }> = {
  foundations: { title: "Foundations", blurb: "Choosing the right learning paradigm for a problem." },
  "classical-ml": { title: "Classical ML", blurb: "Regression, thresholds, neighbors, and clustering." },
  nlp: { title: "NLP Lab", blurb: "Turning text into vectors you can compare." },
  "neural-networks": { title: "Neural Network Lab", blurb: "Forward passes, gradient descent, and backpropagation." },
  transformers: { title: "Transformer Lab", blurb: "Self-attention computed one operation at a time." },
  llm: { title: "LLM Lab", blurb: "From logits to sampled tokens: temperature, top-k, and top-p." },
};

export interface LabMeta { id: string; title: string; area: LabArea; lesson: string; description: string }

export const LABS: LabMeta[] = [
  { id: "paradigm-guide", title: "Which learning paradigm?", area: "foundations", lesson: "learning-paradigms", description: "Answer three questions about a problem and see which paradigm fits, with seven worked scenarios." },
  { id: "linear-regression", title: "Linear regression", area: "classical-ml", lesson: "linear-regression", description: "Drag points, tilt the line, see residuals, MSE and MAE live, and toggle L1/L2 penalties." },
  { id: "threshold", title: "Classification threshold", area: "classical-ml", lesson: "classification-metrics", description: "Move the threshold and watch the confusion matrix, precision, recall, F1, and business cost change." },
  { id: "knn", title: "K-nearest neighbors", area: "classical-ml", lesson: "k-nearest-neighbors", description: "Move a query point, change K and the distance metric, and see what feature scaling does." },
  { id: "kmeans", title: "K-means clustering", area: "classical-ml", lesson: "k-means", description: "Place centroids, step through assignment and update, and change K and the initialization." },
  { id: "tfidf", title: "TF-IDF calculator", area: "nlp", lesson: "text-to-vectors", description: "Edit a corpus and see term frequency, document frequency, IDF, and TF-IDF for every word." },
  { id: "nn-forward", title: "Neural network forward pass", area: "neural-networks", lesson: "neural-network-forward-pass", description: "Change weights and activations in a 2-3-1 network and follow every number and tensor shape." },
  { id: "gradient-descent", title: "Gradient descent", area: "neural-networks", lesson: "gradient-descent", description: "Step across a loss surface with SGD, momentum, or Adam, and push the learning rate until it diverges." },
  { id: "backprop", title: "Backpropagation, step by step", area: "neural-networks", lesson: "backpropagation", description: "A one-neuron classifier: edit the inputs and weights and recompute all nine steps." },
  { id: "attention", title: "Self-attention", area: "transformers", lesson: "self-attention", description: "Step through Q, K, V, QKᵀ, scaling, the causal mask, softmax, and the weighted sum, with shapes." },
  { id: "decoding", title: "Temperature, top-k, top-p", area: "llm", lesson: "decoding", description: "Turn logits into a next-token distribution and watch each sampling knob reshape it." },
];

export const getLab = (id: string) => LABS.find((l) => l.id === id);
