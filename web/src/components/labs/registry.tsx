"use client";
import dynamic from "next/dynamic";
import type { ComponentType } from "react";

// Lab id (see src/lib/labs.ts) -> lazily loaded component. Every id in LABS must appear here (checked by a test).
export const LAB_COMPONENTS: Record<string, ComponentType> = {
  "paradigm-guide": dynamic(() => import("./ParadigmGuideLab")),
  "linear-regression": dynamic(() => import("./LinearRegressionLab")),
  threshold: dynamic(() => import("./ThresholdLab")),
  knn: dynamic(() => import("./KnnLab")),
  kmeans: dynamic(() => import("./KMeansLab")),
  tfidf: dynamic(() => import("./TfidfLab")),
  "nn-forward": dynamic(() => import("./NeuralNetLab")),
  "gradient-descent": dynamic(() => import("./GradientDescentLab")),
  backprop: dynamic(() => import("./BackpropLab")),
  attention: dynamic(() => import("./AttentionLab")),
  decoding: dynamic(() => import("./DecodingLab")),
  "fit-explorer": dynamic(() => import("./FitExplorerLab")),
  "decision-tree": dynamic(() => import("./DecisionTreeLab")),
  boosting: dynamic(() => import("./BoostingLab")),
  pca: dynamic(() => import("./PcaLab")),
  convolution: dynamic(() => import("./ConvolutionLab")),
  tokenizer: dynamic(() => import("./TokenizerLab")),
  rag: dynamic(() => import("./RagLab")),
  architecture: dynamic(() => import("./ArchitectureLab")),
  "prompt-injection": dynamic(() => import("./PromptInjectionLab")),
};
