// Module numbers in teaching order. Modules 24+ were added later, so phases list them explicitly.
export const PHASES = [
  { title: "Foundations", modules: [0, 1, 2], blurb: "Math, information theory, vocabulary, and how to run a valid experiment." },
  { title: "Classical machine learning", modules: [3, 4, 5, 6, 7, 8, 24], blurb: "Regression, classification, neighbors, trees, clustering, PCA, and applied ML: features, imbalance, tuning, time series." },
  { title: "Deep learning", modules: [9, 10, 11, 12, 13, 25, 29, 27], blurb: "Text as vectors, neural networks, PyTorch, training and debugging, architectures, generative models, vision, audio, and training at scale." },
  { title: "Transformers and LLMs", modules: [14, 26, 15, 28, 16, 17], blurb: "Attention and its variants, reinforcement learning for post-training, pretraining, MoE, fine-tuning, prompting, RAG, and evaluation." },
  { title: "LLM inference and agents", modules: [18, 19], blurb: "KV cache, batching, quantization, serving cost, tool calling, memory, MCP, and multi-agent systems." },
  { title: "AI engineering and system design", modules: [20, 21, 22, 23], blurb: "Production systems, security, fairness and privacy, ML system design, monitoring, A/B testing, and projects." },
];
