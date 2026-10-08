export type QuestionType = "single" | "multi" | "numeric" | "fill" | "order" | "match" | "short";

export interface Question {
  id: string;
  type: QuestionType;
  difficulty: "easy" | "medium" | "hard";
  prompt: string;
  options?: string[];
  pairs?: [string, string][];
  answer?: number | number[] | string[];
  tolerance?: number;
  explanation: string;
  why_not?: string[];
  sample_answer?: string;
}

export interface Quiz {
  id: string;
  title: string;
  lesson?: string;
  questions: Question[];
}

/** Grade a response. `short` answers are self-graded, so they return null. */
export function grade(q: Question, response: unknown): boolean | null {
  switch (q.type) {
    case "single":
      return response === q.answer;
    case "multi": {
      const want = [...((q.answer as number[]) ?? [])].sort().join(",");
      return [...((response as number[]) ?? [])].sort().join(",") === want;
    }
    case "numeric": {
      const v = typeof response === "number" ? response : parseFloat(String(response ?? "").replace(/,/g, ""));
      return Number.isFinite(v) && Math.abs(v - (q.answer as number)) <= (q.tolerance ?? 1e-6);
    }
    case "fill": {
      const norm = (s: string) => s.toLowerCase().replace(/\s+/g, "");
      return ((q.answer as string[]) ?? []).some((a) => norm(String(a)) === norm(String(response ?? "")));
    }
    case "order":
      // response: array of option indices in the learner's order; correct order is 0..n-1
      return Array.isArray(response) && response.every((v, i) => v === i) && response.length === q.options?.length;
    case "match":
      // response: array where response[i] = index of the right-hand item chosen for left item i
      return Array.isArray(response) && response.length === q.pairs?.length && response.every((v, i) => v === i);
    case "short":
      return null;
  }
}
