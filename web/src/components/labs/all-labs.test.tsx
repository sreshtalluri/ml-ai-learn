import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import ArchitectureLab from "./ArchitectureLab";
import BoostingLab from "./BoostingLab";
import ConvolutionLab, { conv2d, maxPool } from "./ConvolutionLab";
import DecisionTreeLab from "./DecisionTreeLab";
import FitExplorerLab from "./FitExplorerLab";
import PcaLab from "./PcaLab";
import PromptInjectionLab from "./PromptInjectionLab";
import RagLab from "./RagLab";
import TokenizerLab from "./TokenizerLab";

afterEach(cleanup);

describe("new labs render", () => {
  const labs = { ArchitectureLab, BoostingLab, ConvolutionLab, DecisionTreeLab, FitExplorerLab, PcaLab, PromptInjectionLab, RagLab, TokenizerLab };
  for (const [name, Lab] of Object.entries(labs)) {
    it(name, () => {
      render(<Lab />);
      expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
    });
  }
});

describe("convolution lab math matches the lesson", () => {
  it("vertical-edge kernel on the 5x5 example gives 0, 27, 27", () => {
    const img = Array.from({ length: 5 }, () => [0, 0, 0, 9, 9]);
    const k = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
    expect(conv2d(img, k, 1, 0)[0]).toEqual([0, 27, 27]);
    expect(conv2d(img, k, 1, 1).length).toBe(5);
    expect(conv2d(img, k, 2, 0).length).toBe(2);
  });
  it("max pool matches the worked example", () => {
    expect(maxPool([[1, 3, 2, 1], [4, 6, 5, 0], [1, 2, 9, 8], [3, 1, 4, 7]])).toEqual([[6, 5], [3, 9]]);
  });
});
