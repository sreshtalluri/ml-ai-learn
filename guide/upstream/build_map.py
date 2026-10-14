"""Build guide/upstream/map.json from hand classification. Line numbers = rows of aie-index.tsv (1-based)."""
import json, sys
from pathlib import Path

TMP = Path("/Users/sreshtalluri/.claude/jobs/9b98b61a/tmp")
OUR = {p.stem[3:] for p in (TMP / "repo/guide/lessons").glob("*/[0-9][0-9]-*.md")}
MODS = {p.name for p in (TMP / "repo/guide/lessons").iterdir() if p.is_dir()}
NEW_MODS = {"24-applied-ml", "25-computer-vision", "29-speech-and-audio", "26-reinforcement-learning",
            "27-training-at-scale", "28-prompting-and-context"}

# n|mode|target-or-module|priority|reason   (E=enrich, I=import, L=link, S=skip)
SPEC = """
12|L|python-toolkit|3|Profiling and debugging habits; good Go-deeper for the toolkit lesson.
13|L|python-toolkit|3|Python idioms for AI code; supplementary reading.
14|E|vectors-and-matrices|2|Geometric linear-algebra intuition with from-scratch code.
15|E|vectors-and-matrices|1|Core vector/matrix ops implemented from scratch.
16|E|vectors-and-matrices|2|Matrices as transformations; complements our visuals.
17|E|calculus-for-ml|1|Derivatives/gradients code walkthrough matches our calculus lesson.
18|E|backpropagation|1|Autodiff from scratch is the code heart of backprop.
19|E|probability-and-statistics|1|Distributions with sampling code.
20|E|naive-bayes|2|Bayes theorem worked code feeds Naive Bayes directly.
21|E|gradient-descent|2|Optimization basics overlap gradient descent.
22|I|00-foundations|1|Entropy, cross-entropy, KL: constant interview topic, only passing coverage.
23|E|pca|2|Dimensionality reduction code complements PCA lesson.
24|E|pca|1|SVD from scratch underpins PCA; frequent interview question.
25|E|python-toolkit|2|Tensor ops/broadcasting code fits the NumPy toolkit lesson.
26|I|00-foundations|2|Log-sum-exp, overflow, precision: practical and often asked.
27|E|k-nearest-neighbors|2|Distance metrics code powers KNN.
28|E|probability-and-statistics|1|Hypothesis tests, CIs, bootstrap in code.
29|L|probability-and-statistics|3|MCMC/importance sampling; deeper than interviews need.
30|E|linear-regression|3|Linear systems / normal equations behind least squares.
31|L|support-vector-machines|3|Convexity theory; optional depth for SVM/optimization.
33|L|29-speech-and-audio/spectrograms-mel-features|3|Fourier background for spectrogram lesson.
34|L|ml-system-design|3|Graph basics; useful for graph-feature/GNN follow-ups.
35|L|probability-and-statistics|3|Markov chains/random walks; niche.
36|E|ml-vocabulary|2|Intro framing overlaps our vocabulary lesson.
37|E|linear-regression|1|Linear regression from scratch with gradient descent.
38|E|logistic-regression|1|Logistic regression from scratch.
39|E|decision-trees|1|Tree + forest from scratch.
40|E|support-vector-machines|1|SVM from scratch.
41|E|k-nearest-neighbors|1|KNN from scratch.
42|E|k-means|2|K-means and friends from scratch.
43|I|24-applied-ml|1|Feature engineering is core DS/MLE interview material; not covered.
44|E|classification-metrics|1|Cross-validation and metrics code.
45|E|overfitting-and-bias-variance|1|Bias-variance simulation code.
46|E|random-forests-and-boosting|1|Bagging/boosting from scratch.
47|I|24-applied-ml|1|Grid/random/Bayesian search; standard interview question.
48|E|ml-workflow|2|Pipelines prevent leakage; matches workflow lesson.
49|E|naive-bayes|1|Naive Bayes from scratch.
50|I|24-applied-ml|1|Time series validation and features; big DS interview gap.
51|I|24-applied-ml|2|Anomaly detection methods; common applied problem.
52|I|24-applied-ml|1|Imbalanced data handling asked constantly; only passing coverage.
53|L|24-applied-ml/feature-engineering|3|Overlaps feature engineering import; link as deeper reading.
54|E|ml-system-design|1|Problem-to-deployment framing for ML system design.
55|I|22-ml-system-design|1|Drift and monitoring: MLE interview staple, no dedicated lesson.
56|E|ml-system-design|2|Feature stores and training-serving skew deepen design lesson.
57|E|neural-network-forward-pass|2|Perceptron from scratch.
58|E|neural-network-forward-pass|1|MLP forward pass from scratch.
59|E|backpropagation|1|Backprop from scratch.
60|E|neural-network-forward-pass|2|Activation functions code and comparison.
61|I|10-neural-networks|2|Loss function choice (CE vs MSE, focal) is frequently probed.
62|E|gradient-descent|1|SGD/momentum/Adam from scratch.
63|E|training-and-regularization|1|Dropout, weight decay, etc. in code.
64|E|training-and-regularization|2|Xavier/He init and training stability.
65|E|training-and-regularization|2|LR schedules and warmup.
66|E|backpropagation|1|Mini autograd framework; the best backprop code walkthrough.
67|I|10-neural-networks|2|PyTorch fluency needed for every coding round.
68|L|neural-network-forward-pass|3|JAX alternative; optional.
69|I|12-training-regularization|1|"Loss not decreasing" debugging is a top practical interview question.
70|L|convolutional-networks|3|Pixel/color-space basics; background reading.
71|E|convolutional-networks|1|Convolution from scratch.
72|E|convolutional-networks|1|LeNet to ResNet architecture history with code.
73|E|convolutional-networks|2|End-to-end image classification training.
74|E|autoencoders-diffusion-and-transfer|2|Transfer learning code matches our transfer section.
75|I|25-computer-vision|2|Detection, IoU, NMS, mAP: classic CV interview set.
76|I|25-computer-vision|2|Segmentation/U-Net; core CV task missing.
77|L|25-computer-vision/object-detection-yolo|3|Mask R-CNN depth after detection.
78|S||3|Duplicate of 08-generative-ai/03 GANs.
79|S||3|Duplicate of 08-generative-ai/06 DDPM.
80|S||3|Duplicate of 08-generative-ai/07 latent diffusion.
81|L|25-computer-vision/vision-transformers|3|Video modeling; niche for most roles.
82|L|25-computer-vision/vision-transformers|3|3D vision/NeRF; niche.
83|I|25-computer-vision|2|ViT patch tokens bridge CNNs and transformers.
84|L|serving-llms|3|Edge deployment tips; optional.
85|S||3|Capstone; not surfaced.
86|I|25-computer-vision|3|SimCLR/DINO/MAE: self-supervised learning for applied scientists.
87|S||3|Duplicate of 12-multimodal-ai/02 CLIP.
88|L|25-computer-vision/vision-language-models|3|OCR/doc understanding; useful applied reading.
89|L|vector-search|3|Metric learning/triplet loss behind embedding retrieval.
90|S||3|Pose estimation; niche.
91|S||3|3D Gaussian splatting; niche research.
92|L|13-deep-architectures/flow-matching-rectified-flows|3|DiT/rectified flow follow-up to flow matching.
93|L|25-computer-vision/semantic-segmentation-unet|3|SAM open-vocab segmentation; product-specific.
94|I|25-computer-vision|2|ViT-MLP-LLM VLM pattern; how multimodal LLMs work.
95|S||3|Monocular depth; niche.
96|S||3|Multi-object tracking; niche.
97|L|autoencoders-diffusion-and-transfer|3|World models/video diffusion; frontier reading.
98|E|text-to-vectors|2|Tokenize/stem/lemmatize code.
99|E|text-to-vectors|1|BoW/TF-IDF from scratch.
100|E|word-embeddings|1|Word2Vec from scratch.
101|E|word-embeddings|2|GloVe/fastText/subword embeddings.
102|L|text-to-vectors|3|Sentiment classification example project.
103|I|09-classical-nlp|3|NER/sequence labeling; classic applied-scientist NLP question.
104|L|text-to-vectors|3|POS/parsing; dated for most interviews.
105|E|recurrent-networks|2|CNN/RNN text classifiers code.
106|E|recurrent-networks|2|Seq2seq encoder-decoder code.
107|E|self-attention|1|Bahdanau-to-transformer attention motivation and code.
108|L|transformer-architecture|3|Machine translation application.
109|L|rag-pipeline|3|Summarization techniques.
110|L|rag-pipeline|3|QA systems pre-LLM.
111|E|vector-search|2|IR/BM25 fundamentals in code.
112|I|09-classical-nlp|3|LDA topic modeling still asked in DS interviews.
113|E|tokenization-and-pretraining|2|N-gram LMs and perplexity ground language modeling.
114|S||3|Chatbot history; low value.
115|L|tokenization-and-pretraining|3|Multilingual NLP considerations.
116|E|tokenization-and-pretraining|1|BPE/WordPiece/Unigram compared.
117|E|decoding|1|Constrained decoding mechanics.
118|L|llm-evaluation|3|NLI; used in faithfulness checks.
119|E|vector-search|2|Modern embedding models deep dive.
120|E|rag-pipeline|1|Chunking strategies for RAG.
121|S||3|Coreference; niche.
122|S||3|Entity linking; niche.
123|L|rag-pipeline|3|KG construction for GraphRAG.
124|E|llm-evaluation|1|RAGAS/G-Eval metric mechanics.
125|L|llm-evaluation|3|Long-context benchmarks.
126|S||3|Dialogue state tracking; dated.
127|L|29-speech-and-audio/spectrograms-mel-features|3|Audio basics prerequisite reading.
128|I|29-speech-and-audio|3|Spectrograms/MFCCs: how audio becomes model input.
129|L|29-speech-and-audio/spectrograms-mel-features|3|Audio classification example.
130|I|29-speech-and-audio|3|CTC and ASR; durable sequence-alignment idea.
131|L|29-speech-and-audio/speech-recognition-asr|3|Whisper as the modern ASR example.
132|S||3|Speaker verification; niche.
133|L|29-speech-and-audio/speech-recognition-asr|3|TTS overview.
134|S||3|Voice cloning; niche.
135|S||3|Music generation; niche.
136|S||3|Audio-LMs; fast-moving product coverage.
137|S||3|Real-time audio engineering; niche.
138|S||3|Capstone; not surfaced.
139|S||3|Neural codecs; niche.
140|S||3|VAD; niche.
141|S||3|Moshi full-duplex; research niche.
142|S||3|Anti-spoofing; niche.
143|L|29-speech-and-audio/speech-recognition-asr|3|WER/MOS audio evaluation metrics.
144|E|self-attention|2|Why RNNs fall short; motivation for attention.
145|E|self-attention|1|Self-attention from scratch.
146|E|self-attention|1|Multi-head attention code.
147|E|transformer-architecture|1|Sinusoidal/RoPE/ALiBi positional encodings.
148|E|transformer-architecture|1|Full encoder-decoder transformer in code.
149|I|14-transformers|1|Encoder-only/MLM and BERT vs GPT is a top interview question.
150|E|tokenization-and-pretraining|1|Causal LM objective and GPT in code.
151|L|14-transformers/bert-masked-language-modeling|3|T5/BART encoder-decoder follow-up.
152|S||3|Duplicate of 04-computer-vision/14 ViT.
153|S||3|Duplicate of Whisper lesson.
154|I|15-llms|1|MoE routing powers frontier models; asked frequently.
155|E|llm-inference|1|KV cache and FlashAttention mechanics.
156|I|15-llms|2|Chinchilla scaling laws; compute-optimal training questions.
157|E|transformer-architecture|1|Build-a-transformer capstone code.
158|I|14-transformers|1|MQA/GQA/sliding-window attention; common system-level question.
159|E|llm-inference|2|Speculative decoding mechanics.
160|L|autoencoders-diffusion-and-transfer|3|Generative model taxonomy overview.
161|E|autoencoders-diffusion-and-transfer|1|VAE from scratch.
162|I|13-deep-architectures|2|GANs: adversarial training still a standard question.
163|L|13-deep-architectures/gans-generator-discriminator|3|Conditional GANs follow-up.
164|S||3|StyleGAN; dated niche.
165|E|autoencoders-diffusion-and-transfer|1|DDPM from scratch.
166|I|13-deep-architectures|2|Latent diffusion: how Stable Diffusion actually works.
167|L|fine-tuning-in-practice|3|ControlNet/LoRA conditioning for image models.
168|S||3|Image editing; niche.
169|S||3|Video generation; product-driven.
170|S||3|Audio generation; niche.
171|S||3|3D generation; niche.
172|I|13-deep-architectures|3|Flow matching: modern generative training objective.
173|L|autoencoders-diffusion-and-transfer|3|FID/CLIP score evaluation.
174|S||3|VAR; research niche.
175|I|26-reinforcement-learning|2|MDPs: RL foundation needed before RLHF.
176|L|26-reinforcement-learning/mdps-states-actions-rewards|3|Policy/value iteration depth.
177|L|26-reinforcement-learning/q-learning-sarsa|3|Monte Carlo methods depth.
178|I|26-reinforcement-learning|2|TD learning, Q-learning vs SARSA: classic interview pair.
179|I|26-reinforcement-learning|2|DQN: replay buffer and target networks.
180|I|26-reinforcement-learning|2|REINFORCE: policy gradients underpin RLHF.
181|L|26-reinforcement-learning/ppo|3|Actor-critic bridge to PPO.
182|I|26-reinforcement-learning|2|PPO: the RLHF optimizer.
183|E|adapting-llms|1|Reward modeling for RLHF.
184|S||3|Multi-agent RL; niche.
185|S||3|Sim-to-real; robotics niche.
186|L|26-reinforcement-learning/ppo|3|AlphaZero to reasoning-era RL.
187|E|tokenization-and-pretraining|1|Tokenizer algorithms compared.
188|E|tokenization-and-pretraining|1|Tokenizer built from scratch.
189|E|tokenization-and-pretraining|2|Pretraining data pipelines: dedup, filtering.
190|E|tokenization-and-pretraining|1|Mini GPT pretraining loop.
191|I|27-training-at-scale|1|DDP/FSDP/ZeRO and memory math; key MLE/LLM interview topic.
192|E|fine-tuning-in-practice|1|SFT walkthrough in code.
193|E|adapting-llms|1|RLHF reward model + PPO.
194|E|adapting-llms|1|DPO derivation and code.
195|L|adapting-llms|3|Constitutional AI / self-improvement.
196|E|llm-evaluation|2|Benchmarks and LM harness.
197|E|llm-inference|1|Quantization from scratch.
198|E|llm-inference|2|Inference optimization techniques.
199|S||3|Pipeline capstone; duplicates other lessons.
200|I|15-llms|1|Read model configs, compute params and KV memory: high-signal skill.
201|L|llm-inference|3|EAGLE-3 speculative decoding depth.
202|L|14-transformers/attention-variants|3|Differential attention research.
203|L|14-transformers/attention-variants|3|Native sparse attention research.
204|L|15-llms/open-models-architecture-walkthroughs|3|Multi-token prediction.
205|S||3|DualPipe; very specialized.
206|L|15-llms/open-models-architecture-walkthroughs|3|DeepSeek-V3 walkthrough as case study.
207|L|15-llms/open-models-architecture-walkthroughs|3|Hybrid SSM-transformer.
208|S||3|Async inference research niche.
209|S||3|Duplicate of speculative decoding lessons.
210|I|27-training-at-scale|2|Activation recomputation memory trade-off.
211|I|28-prompting-and-context|1|Prompt engineering patterns; AI-engineer interview core.
212|I|28-prompting-and-context|2|Few-shot, CoT, ToT reasoning prompts.
213|E|decoding|2|Structured outputs and schema validation.
214|E|vector-search|2|Embeddings code for vector search.
215|I|28-prompting-and-context|1|Context budgets/memory; core AI-engineer system skill.
216|E|rag-pipeline|1|Basic RAG built in code.
217|E|rag-pipeline|1|Hybrid search, reranking, chunking in code.
218|E|fine-tuning-in-practice|1|LoRA/QLoRA walkthrough.
219|E|tool-use-and-agents|1|Function calling code.
220|E|llm-evaluation|1|Testing LLM apps code.
221|E|reliability-cost-and-observability|1|Caching, rate limits, cost control.
222|E|ai-security|1|Guardrails and content filtering.
223|E|production-architecture|1|Production LLM app walkthrough.
224|I|19-agents|2|MCP is now standard agent tooling vocabulary.
225|E|reliability-cost-and-observability|2|Prompt caching economics.
226|L|reliable-agents|3|Graph state machines for agents.
227|L|tool-use-and-agents|3|Framework trade-offs.
228|S||3|Duplicate of ViT import.
229|I|25-computer-vision|1|CLIP contrastive learning underlies multimodal and retrieval.
230|L|25-computer-vision/vision-language-models|3|BLIP-2 Q-Former bridge.
231|L|25-computer-vision/vision-language-models|3|Flamingo gated cross-attention.
232|L|25-computer-vision/vision-language-models|3|LLaVA visual instruction tuning.
233|S||3|Any-resolution vision; niche.
234|L|25-computer-vision/vision-language-models|3|VLM training recipes.
235|S||3|Model-family specific.
236|S||3|Model-family specific.
237|S||3|Model-family specific.
238|S||3|Early-fusion research niche.
239|S||3|Research niche.
240|S||3|Research niche.
241|S||3|Research niche.
242|S||3|Research niche.
243|S||3|Research niche.
244|S||3|Video-LM niche.
245|S||3|Long-video niche.
246|S||3|Audio-LM niche.
247|S||3|Omni-model product coverage.
248|S||3|Robotics VLAs; niche.
249|L|rag-pipeline|3|Document/diagram understanding for doc RAG.
250|L|rag-pipeline|3|ColPali vision-native retrieval.
251|L|rag-pipeline|3|Multimodal RAG.
252|L|tool-use-and-agents|3|Computer-use agents.
253|E|tool-use-and-agents|1|Tool interface design with code.
254|E|tool-use-and-agents|2|Provider function-calling comparison.
255|L|tool-use-and-agents|3|Parallel/streaming tool calls.
256|S||3|Duplicate of 11-llm-engineering/03 structured outputs.
257|E|tool-use-and-agents|2|Tool schema design guidance.
258|L|19-agents/model-context-protocol|3|MCP protocol details.
259|L|19-agents/model-context-protocol|3|Building an MCP server.
260|S||3|MCP client internals; product depth.
261|S||3|MCP transports; product depth.
262|S||3|MCP spec detail.
263|S||3|MCP spec detail.
264|S||3|MCP spec detail.
265|S||3|MCP spec detail.
266|S||3|MCP spec detail.
267|E|ai-security|2|Tool poisoning attacks on agents.
268|S||3|MCP auth spec detail.
269|S||3|MCP gateway product detail.
270|S||3|MCP auth spec detail.
271|L|19-agents/why-multi-agent|3|A2A protocol.
272|E|reliability-cost-and-observability|2|OTel tracing for LLM calls.
273|L|production-architecture|3|LLM routing layer products.
274|L|tool-use-and-agents|3|Agent skills concept.
275|S||3|Capstone; not surfaced.
276|S||3|Skills product detail.
277|S||3|Skills product detail.
278|S||3|Skills product detail.
279|S||3|Skills product detail.
280|S||3|MCP spec detail.
281|S||3|MCP spec detail.
282|S||3|MCP ops detail.
283|S||3|MCP ops detail.
284|E|tool-use-and-agents|1|Agent loop from scratch.
285|E|tool-use-and-agents|2|ReWOO/plan-and-execute pattern code.
286|L|reliable-agents|3|Reflexion research pattern.
287|L|reliable-agents|3|Tree of Thoughts / LATS.
288|E|reliable-agents|2|Self-refine/critic loops.
289|S||3|Duplicate of function calling lessons.
290|I|19-agents|1|Agent memory design asked in every AI-engineer loop.
291|L|19-agents/memory-virtual-context-memgpt|3|Memory blocks follow-up.
292|L|19-agents/memory-virtual-context-memgpt|3|Hybrid vector/graph memory.
293|S||3|Voyager research niche.
294|S||3|HTN planning niche.
295|E|reliable-agents|1|Workflow patterns: simple over complex.
296|S||3|Duplicate of 11-llm-engineering/16.
297|S||3|Framework-specific.
298|S||3|Framework-specific.
299|S||3|Vendor SDK.
300|S||3|Vendor SDK.
301|S||3|Vendor runtimes.
302|L|llm-evaluation|3|SWE-bench/GAIA agent benchmarks.
303|S||3|Benchmark detail.
304|S||3|Vendor computer-use comparison.
305|S||3|Vendor voice frameworks.
306|S||3|Duplicate of OTel lesson.
307|S||3|Vendor observability tools.
308|L|19-agents/why-multi-agent|3|Multi-agent debate.
309|E|reliable-agents|1|Agent failure modes catalog.
310|E|ai-security|1|Prompt injection defenses.
311|L|19-agents/why-multi-agent|3|Orchestration patterns summary.
312|L|production-architecture|3|Queue/event/cron agent runtimes.
313|E|llm-evaluation|1|Eval-driven agent development.
314|S||3|Coding-agent workbench; tooling-specific.
315|S||3|Workbench tooling.
316|S||3|Workbench tooling.
317|S||3|Workbench tooling.
318|S||3|Workbench tooling.
319|S||3|Workbench tooling.
320|S||3|Workbench tooling.
321|S||3|Workbench tooling.
322|S||3|Workbench tooling.
323|S||3|Workbench tooling.
324|S||3|Workbench tooling.
325|S||3|Capstone; not surfaced.
326|S||3|Agent-coding workflow; not ML.
327|S||3|Agent-coding workflow; not ML.
328|S||3|Agent-coding workflow; not ML.
329|S||3|Agent-coding workflow; not ML.
330|S||3|Product practice; not ML.
331|S||3|Product practice; not ML.
332|S||3|Product practice; not ML.
333|S||3|Product practice; not ML.
334|S||3|Product practice; not ML.
335|L|ml-system-design|3|Success metrics design before results.
336|S||3|Product practice; not ML.
337|S||3|Product practice; not ML.
338|L|reliable-agents|3|Long-horizon agent overview.
339|L|adapting-llms|3|STaR self-taught reasoning.
340|S||3|Research system.
341|S||3|Research system.
342|S||3|Research system.
343|S||3|Research policy topic.
344|S||3|Speculative safety topic.
345|S||3|Speculative safety topic.
346|S||3|Product landscape; dates fast.
347|S||3|Product-specific permission modes.
348|S||3|Browser agent products.
349|L|reliable-agents|3|Durable execution for long agents.
350|E|reliable-agents|2|Budgets, iteration caps, cost governors.
351|L|reliable-agents|3|Kill switches and circuit breakers.
352|E|reliable-agents|2|Human-in-the-loop propose-then-commit.
353|L|reliable-agents|3|Checkpoints and rollback.
354|S||3|Duplicate of constitutional AI.
355|S||3|Duplicate of moderation lesson.
356|S||3|Policy document; not interview.
357|S||3|Policy document; not interview.
358|L|llm-evaluation|3|METR time horizons evaluation.
359|S||3|Policy topic.
360|I|19-agents|2|When to go multi-agent and orchestration trade-offs.
361|S||3|Historical protocols.
362|S||3|Protocol detail.
363|S||3|Abstract model; covered by import.
364|L|19-agents/why-multi-agent|3|Supervisor pattern code.
365|S||3|Covered by multi-agent import.
366|S||3|Duplicate of debate lesson.
367|L|19-agents/why-multi-agent|3|Role specialization.
368|S||3|Covered by multi-agent import.
369|S||3|Framework pattern detail.
370|L|19-agents/why-multi-agent|3|Handoffs and routines.
371|S||3|Duplicate of A2A lesson.
372|S||3|Niche pattern.
373|S||3|BFT for agents; niche.
374|L|28-prompting-and-context/few-shot-cot|3|Self-consistency voting.
375|S||3|Negotiation; niche.
376|S||3|Simulation research.
377|S||3|Research niche.
378|S||3|Research niche.
379|S||3|MARL niche.
380|S||3|Speculative economics.
381|S||3|Covered by production lessons.
382|L|reliable-agents|3|Multi-agent failure modes (MAST).
383|S||3|Benchmark detail.
384|S||3|Dated case studies.
385|S||3|Vendor platforms.
386|S||3|Vendor pricing comparison.
387|L|serving-llms|3|GPU autoscaling on Kubernetes.
388|E|serving-llms|1|PagedAttention and continuous batching internals.
389|S||3|Duplicate of speculative decoding.
390|E|serving-llms|2|Prefix caching / RadixAttention.
391|S||3|Hardware-vendor specific.
392|E|serving-llms|1|TTFT/TPOT/goodput metrics.
393|E|llm-inference|2|AWQ/GPTQ/GGUF production quantization.
394|L|serving-llms|3|Cold-start mitigation.
395|S||3|Multi-region ops niche.
396|L|llm-inference|3|Edge inference targets.
397|S||3|Vendor tool selection.
398|E|reliability-cost-and-observability|2|Semantic caching economics.
399|L|reliability-cost-and-observability|3|Batch APIs for cost.
400|E|reliability-cost-and-observability|2|Model routing for cost.
401|L|serving-llms|3|Disaggregated prefill/decode.
402|S||3|Vendor stack detail.
403|L|production-architecture|3|AI gateways.
404|E|production-architecture|2|Shadow/canary rollouts.
405|E|experimentation-and-ab-testing|1|A/B testing LLM features.
406|L|serving-llms|3|Load testing LLM APIs.
407|S||3|Ops niche.
408|S||3|Ops niche.
409|L|ai-security|3|Secrets and audit logs.
410|S||3|Compliance/certification specific.
411|L|reliability-cost-and-observability|3|LLM unit economics.
412|L|serving-llms|3|Choosing a serving engine.
413|S||3|Demo UI tooling.
414|S||3|Alignment framing; low interview value.
415|L|adapting-llms|3|Reward hacking and Goodhart.
416|E|adapting-llms|2|DPO family variants.
417|L|adapting-llms|3|Sycophancy from RLHF.
418|S||3|Duplicate of constitutional AI.
419|S||3|Alignment research theory.
420|S||3|Alignment research.
421|S||3|Alignment research.
422|S||3|Alignment research.
423|S||3|Alignment research.
424|S||3|Alignment research.
425|E|ai-security|2|Automated red-teaming (PAIR).
426|L|ai-security|3|Many-shot jailbreaking.
427|S||3|Jailbreak niche.
428|E|ai-security|1|Indirect prompt injection attack surface.
429|L|ai-security|3|Red-team tooling.
430|S||3|Dual-use eval niche.
431|S||3|Policy frameworks.
432|S||3|Research program.
433|L|21-safety-security/fairness-criteria-group-individual-counterfactual|3|Bias and representational harm.
434|I|21-safety-security|2|Fairness metrics asked in DS/applied-scientist loops.
435|I|21-safety-security|3|Differential privacy fundamentals.
436|L|ai-security|3|Watermarking/provenance.
437|S||3|Regulation; changes fast.
438|L|ai-security|3|EchoLeak case study.
439|L|ml-system-design|3|Model/dataset cards.
440|S||3|Governance niche.
441|S||3|Ecosystem listing.
442|E|ai-security|2|Moderation classifiers.
443|S||3|Dual-use policy.
444|L|project-ladder|3|Coding-agent capstone idea.
445|L|project-ladder|3|RAG-over-codebase capstone idea.
446|S||3|Capstone; not surfaced.
447|S||3|Capstone; not surfaced.
448|S||3|Capstone; not surfaced.
449|S||3|Capstone; not surfaced.
450|L|project-ladder|3|End-to-end fine-tuning capstone idea.
451|L|project-ladder|3|Production RAG capstone idea.
452|S||3|Capstone; not surfaced.
453|S||3|Capstone; not surfaced.
454|S||3|Capstone; not surfaced.
455|S||3|Capstone; not surfaced.
456|S||3|Capstone; not surfaced.
457|S||3|Capstone; not surfaced.
458|S||3|Capstone; not surfaced.
459|S||3|Capstone; not surfaced.
460|S||3|Capstone; not surfaced.
461|S||3|Harness capstone step.
462|S||3|Harness capstone step.
463|S||3|Harness capstone step.
464|S||3|Harness capstone step.
465|S||3|Harness capstone step.
466|S||3|Harness capstone step.
467|S||3|Harness capstone step.
468|S||3|Harness capstone step.
469|S||3|Harness capstone step.
470|S||3|Harness capstone step.
471|E|tokenization-and-pretraining|1|Clean BPE-from-scratch code.
472|E|tokenization-and-pretraining|2|Sliding-window dataset for next-token training.
473|E|transformer-architecture|2|Token + positional embeddings code.
474|E|self-attention|2|Multi-head self-attention module code.
475|E|transformer-architecture|1|Transformer block from scratch.
476|E|transformer-architecture|2|GPT model assembly.
477|E|tokenization-and-pretraining|2|Training loop and eval.
478|L|fine-tuning-in-practice|3|Loading pretrained weights.
479|E|fine-tuning-in-practice|2|Classifier fine-tuning by head swap.
480|S||3|Duplicate of SFT lesson.
481|E|adapting-llms|2|DPO from scratch code.
482|S||3|Capstone step.
483|S||3|Data download tooling.
484|S||3|Storage tooling.
485|E|training-and-regularization|2|Cosine LR with warmup code.
486|I|27-training-at-scale|2|Mixed precision and gradient clipping; standard training knowledge.
487|E|fine-tuning-in-practice|2|Gradient accumulation code.
488|S||3|Checkpoint tooling.
489|L|27-training-at-scale/scaling-distributed|3|DDP/FSDP from scratch code.
490|L|llm-evaluation|3|LM eval harness.
491|S||3|Research-agent capstone step.
492|S||3|Research-agent capstone step.
493|S||3|Research-agent capstone step.
494|S||3|Research-agent capstone step.
495|S||3|Research-agent capstone step.
496|S||3|Research-agent capstone step.
497|S||3|Research-agent capstone step.
498|S||3|Research-agent capstone step.
499|S||3|VLM capstone step.
500|S||3|VLM capstone step.
501|S||3|VLM capstone step.
502|S||3|VLM capstone step.
503|S||3|VLM capstone step.
504|S||3|VLM capstone step.
505|E|rag-pipeline|2|Chunking strategies compared in code.
506|E|vector-search|1|Hybrid BM25 + dense with RRF.
507|E|rag-pipeline|1|Cross-encoder reranker code.
508|E|rag-pipeline|2|HyDE / multi-query rewriting.
509|E|llm-evaluation|1|Retrieval metrics: MRR, nDCG, faithfulness.
510|S||3|Capstone step.
511|S||3|Capstone step.
512|E|llm-evaluation|2|BLEU/ROUGE classical metrics code.
513|E|llm-evaluation|2|pass@k code execution metric.
514|E|llm-evaluation|2|Perplexity and calibration (ECE).
515|L|llm-evaluation|3|Leaderboard aggregation.
516|S||3|Capstone step.
517|I|27-training-at-scale|2|Ring all-reduce from scratch; distributed fundamentals.
518|S||3|Covered by distributed import.
519|I|27-training-at-scale|2|ZeRO sharding stages; memory math interview favorite.
520|L|27-training-at-scale/scaling-distributed|3|Pipeline parallel bubbles.
521|S||3|Checkpoint tooling.
522|S||3|Capstone step.
523|L|ai-security|3|Jailbreak taxonomy.
524|E|ai-security|2|Prompt-injection detector code.
525|S||3|Capstone step.
526|S||3|Capstone step.
527|S||3|Capstone step.
528|S||3|Capstone step.
"""

SETUP_SKIP = "Setup/tooling trivia; not ML understanding."

def main():
    rows = [l.split("\t") for l in (TMP / "aie-index.tsv").read_text().splitlines() if l.strip()]
    spec = {}
    for line in SPEC.strip().splitlines():
        n, mode, tgt, pri, reason = line.split("|", 4)
        assert int(n) not in spec, n
        spec[int(n)] = (mode, tgt, int(pri), reason)
    imports = {f"{v[1]}/{rows[k-1][0].split('/')[-1][3:]}" for k, v in spec.items() if v[0] == "I"}
    modes = {"E": "enrich", "I": "import", "L": "link", "S": "skip"}
    out, errs = [], []
    for i, r in enumerate(rows, 1):
        path, title = r[0], r[1]
        mode, tgt, pri, reason = spec.get(i, ("S", "", 3, SETUP_SKIP if "00-setup" in path else "Niche; low interview value."))
        if i not in spec and "00-setup" not in path and i not in (32,):
            errs.append(f"unclassified {i} {path}")
        e = {"upstream": path, "title": title, "mode": modes[mode]}
        if mode == "E":
            if tgt not in OUR: errs.append(f"bad enrich target {i} {tgt}")
            e["target"] = tgt
        elif mode == "I":
            if tgt not in MODS | NEW_MODS: errs.append(f"bad module {i} {tgt}")
            e["target_module"] = tgt
        elif mode == "L":
            if tgt not in OUR and tgt not in MODS | NEW_MODS and tgt not in imports:
                errs.append(f"bad link target {i} {tgt}")
            e["target"] = tgt
        e["priority"] = pri
        e["reason"] = reason if i != 32 else "Complex numbers; tangential to ML interviews."
        if len(e["reason"].split()) > 15: errs.append(f"long reason {i}")
        out.append(e)
    if errs:
        sys.exit("\n".join(errs))
    dst = TMP / "repo/guide/upstream/map.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    assert len(json.loads(dst.read_text())) == 528 == len(rows)
    from collections import Counter
    print(Counter(e["mode"] for e in out), Counter((e["mode"], e["priority"]) for e in out if e["mode"] == "import"))

if __name__ == "__main__":
    main()
