import { act, cleanup, render } from "@testing-library/react";
import type { ComponentType } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { getExplainers, getLessons } from "@/lib/content";
import { LABS } from "@/lib/labs";
import AbTestLab from "./AbTestLab";
import AgentLoopLab from "./AgentLoopLab";
import ArchitectureLab from "./ArchitectureLab";
import AttentionLab from "./AttentionLab";
import BackpropLab from "./BackpropLab";
import BatchingLab from "./BatchingLab";
import BoostingLab from "./BoostingLab";
import ConvolutionLab from "./ConvolutionLab";
import DecisionTreeLab from "./DecisionTreeLab";
import DecodingLab from "./DecodingLab";
import FitExplorerLab from "./FitExplorerLab";
import GradientDescentLab from "./GradientDescentLab";
import KMeansLab from "./KMeansLab";
import KnnLab from "./KnnLab";
import KvCacheLab from "./KvCacheLab";
import LinearRegressionLab from "./LinearRegressionLab";
import LoraLab from "./LoraLab";
import NeuralNetLab from "./NeuralNetLab";
import ParadigmGuideLab from "./ParadigmGuideLab";
import PcaLab from "./PcaLab";
import PromptInjectionLab from "./PromptInjectionLab";
import RagLab from "./RagLab";
import RecsysFunnelLab from "./RecsysFunnelLab";
import ThresholdLab from "./ThresholdLab";
import TfidfLab from "./TfidfLab";
import TokenizerLab from "./TokenizerLab";
import VectorSearchLab from "./VectorSearchLab";
import { TourContext } from "./tour";

// Static twin of registry.tsx (which loads labs lazily).
const LAB: Record<string, ComponentType> = {
  "paradigm-guide": ParadigmGuideLab, "linear-regression": LinearRegressionLab, threshold: ThresholdLab, knn: KnnLab, kmeans: KMeansLab,
  tfidf: TfidfLab, "nn-forward": NeuralNetLab, "gradient-descent": GradientDescentLab, backprop: BackpropLab, attention: AttentionLab,
  decoding: DecodingLab, "fit-explorer": FitExplorerLab, "decision-tree": DecisionTreeLab, boosting: BoostingLab, pca: PcaLab,
  convolution: ConvolutionLab, tokenizer: TokenizerLab, rag: RagLab, architecture: ArchitectureLab, "prompt-injection": PromptInjectionLab,
  lora: LoraLab, "vector-search": VectorSearchLab, "kv-cache": KvCacheLab, batching: BatchingLab, "agent-loop": AgentLoopLab,
  "recsys-funnel": RecsysFunnelLab, "ab-test": AbTestLab,
};

const stepsOf = (id: string) => {
  const Lab = LAB[id];
  const { container } = render(<Lab />);
  const attr = container.querySelector("section[data-tour-steps]")?.getAttribute("data-tour-steps") ?? "";
  cleanup();
  return attr.split(" ").filter(Boolean);
};

afterEach(cleanup);

describe("guided tours", () => {
  it("cover every lab", () => expect(Object.keys(LAB).sort()).toEqual(LABS.map((l) => l.id).sort()));

  for (const id of Object.keys(LAB)) {
    it(`${id} has a 4-8 step tour with unique ids`, () => {
      const steps = stepsOf(id);
      expect(steps.length).toBeGreaterThanOrEqual(4);
      expect(steps.length).toBeLessThanOrEqual(8);
      expect(new Set(steps).size).toBe(steps.length);
    });

    it(`${id} survives every tour step being driven by an explainer`, () => {
      const Lab = LAB[id];
      for (const step of stepsOf(id)) {
        const { unmount } = render(<TourContext.Provider value={{ step }}><Lab /></TourContext.Provider>);
        act(() => {}); // flush the effect that applies the step
        unmount();
      }
    });
  }
});

describe("visual explainers", () => {
  const lessons = new Set(getLessons().map((l) => l.slug));
  for (const e of getExplainers()) {
    it(`${e.id} points at a real lab and lesson, and every step exists in the lab's tour`, () => {
      expect(LAB[e.lab], e.lab).toBeDefined();
      expect(lessons.has(e.lesson), e.lesson).toBe(true);
      const tour = stepsOf(e.lab);
      const used = e.sections.map((s) => s.step).filter((s): s is string => !!s);
      expect(used.length).toBeGreaterThanOrEqual(4);
      for (const s of used) expect(tour, `${e.id} uses step ${s}`).toContain(s);
    });
  }
});
