import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { actions, emptyProgress, parseProgress, STORAGE_KEY, streak, useProgress } from "./progress";

beforeEach(() => window.localStorage.clear());

describe("progress persistence", () => {
  it("persists lesson completion to localStorage and notifies subscribers", () => {
    const { result } = renderHook(() => useProgress());
    expect(result.current.lessons).toEqual({});
    act(() => actions.setLessonComplete("linear-regression", true));
    expect(result.current.lessons["linear-regression"]).toBeTypeOf("number");
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY)!).lessons["linear-regression"]).toBeTypeOf("number");
    act(() => actions.setLessonComplete("linear-regression", false));
    expect(result.current.lessons["linear-regression"]).toBeUndefined();
  });

  it("keeps the best quiz score and the latest wrong answers", () => {
    actions.recordQuiz("q", 3, 4, ["a"]);
    actions.recordQuiz("q", 1, 4, ["a", "b", "c"]);
    const p = parseProgress(window.localStorage.getItem(STORAGE_KEY)!);
    expect(p.quizzes.q).toMatchObject({ best: 0.75, last: 0.25, attempts: 2, wrong: ["a", "b", "c"] });
  });

  it("round-trips export and import", () => {
    actions.toggleBookmark("/learn/k-means/", "K-means");
    actions.setNote("/learn/k-means/", "scale features first");
    const dump = actions.exportJSON();
    actions.reset();
    expect(parseProgress(window.localStorage.getItem(STORAGE_KEY)!).bookmarks).toEqual([]);
    actions.importJSON(dump);
    const p = parseProgress(window.localStorage.getItem(STORAGE_KEY)!);
    expect(p.bookmarks[0].title).toBe("K-means");
    expect(p.notes["/learn/k-means/"]).toBe("scale features first");
  });

  it("rejects files that are not progress exports", () => {
    expect(() => actions.importJSON("{}")).toThrow(/progress file/);
    expect(() => actions.importJSON(JSON.stringify({ version: 1, lessons: [] }))).toThrow(/lessons/);
    expect(() => actions.importJSON("not json")).toThrow();
  });

  it("fills in fields missing from older exports", () => {
    expect(parseProgress(JSON.stringify({ version: 1 }))).toEqual(emptyProgress());
  });

  it("recent list dedupes and caps at 8", () => {
    for (let i = 0; i < 10; i++) actions.visit(`/p${i}/`, `P${i}`);
    actions.visit("/p3/", "P3");
    const p = parseProgress(window.localStorage.getItem(STORAGE_KEY)!);
    expect(p.recent).toHaveLength(8);
    expect(p.recent[0].path).toBe("/p3/");
    expect(p.recent.filter((r) => r.path === "/p3/")).toHaveLength(1);
  });
});

describe("streak", () => {
  const today = new Date("2026-10-07T12:00:00Z");
  it("counts consecutive days ending today or yesterday", () => {
    expect(streak(["2026-10-05", "2026-10-06", "2026-10-07"], today)).toBe(3);
    expect(streak(["2026-10-05", "2026-10-06"], today)).toBe(2);
    expect(streak(["2026-10-01"], today)).toBe(0);
  });
});
