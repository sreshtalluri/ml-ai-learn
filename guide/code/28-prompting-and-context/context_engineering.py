# Adapted from AI Engineering from Scratch (MIT), phases/11-llm-engineering/05-context-engineering
"""Context engineering: a token budget, relevance-ordered packing, history compaction.

Token counts use the rough 1.3 tokens per English word rule; real counts depend on
the tokenizer. Component sizes, documents, and conversation are SYNTHETIC. No LLM is called: the point
is what you send. A real client would receive `assemble(...)` as its messages.

Run from guide/:  uv run code/28-prompting-and-context/context_engineering.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def count_tokens(text):
    """Rough estimate. Use the provider's token-count endpoint or tokenizer in production."""
    return int(len(text.split()) * 1.3)


# ---------- 1. the budget ----------
class ContextBudget:
    def __init__(self, window, reserve_output):
        self.available = window - reserve_output       # output tokens share the window
        self.used = {}

    def add(self, name, text, cap=None):
        limit = min(cap or 10**9, self.available - sum(self.used.values()))
        words = text.split()[: int(limit / 1.3)]         # keep the words that fit (1.3 tokens/word)
        self.used[name] = count_tokens(" ".join(words))
        return " ".join(words)

    def remaining(self):
        return self.available - sum(self.used.values())


# ---------- 2. relevance scoring and lost-in-the-middle ordering ----------
def score_relevance(query, docs):
    q = set(query.lower().split())
    return [len(q & set(d.lower().split())) / len(q) for d in docs]


def edges_first(items, scores):
    """Best item first, second best last, weakest in the middle."""
    ranked = [it for _, it in sorted(zip(scores, items), key=lambda p: -p[0])]
    return ranked[::2] + ranked[1::2][::-1]


# ---------- 3. conversation compaction ----------
def compact_history(turns, max_tokens, keep_recent=4, summarize=None):
    summarize = summarize or (lambda ts: "Summary of earlier turns: " + "; ".join(t[:40] for t in ts))
    total = sum(count_tokens(t) for t in turns)
    if total <= max_tokens:
        return turns
    old, recent = turns[:-keep_recent], turns[-keep_recent:]
    return [summarize(old)] + recent                     # a real system asks an LLM for the summary


# ---------- 4. dynamic tool selection ----------
TOOLS = {f"{area}_{verb}": area for area in ["files", "git", "calendar", "email", "db", "web"]
         for verb in ["read", "write", "search", "list", "delete"]}
TOOL_TOKENS = 150                                        # assumed schema size per tool


def select_tools(query):
    hits = {area for area in set(TOOLS.values()) if area in query.lower() or (area == "files" and "file" in query)}
    return [t for t, area in TOOLS.items() if area in hits]


# ---------- 5. per-call input over a conversation ----------
NAIVE = {"system": 500, "tools": 30 * TOOL_TOKENS, "docs": 8 * 500, "query": 200}
LEAN = {"system": 500, "tools": 10 * TOOL_TOKENS, "docs": 3 * 500, "query": 200}
TURN, KEEP, SUMMARY = 250, 4, 150


def input_tokens(t, lean):
    """Input tokens for call t (1-based): fixed parts plus the history before it."""
    if not lean:
        return sum(NAIVE.values()) + TURN * (t - 1)
    hist = TURN * (t - 1) if t - 1 <= KEEP else TURN * KEEP + SUMMARY
    return sum(LEAN.values()) + hist


def main():
    print("=== 1. Budget: a 128,000-token window ===")
    window, reserve = 128_000, 4_000
    parts = {"system prompt": 500, "tool schemas (30 x 150)": 30 * TOOL_TOKENS, "retrieved docs (8 x 500)": 4_000,
             "history (12 turns x 250)": 3_000, "user query": 200}
    used = sum(parts.values())
    for k, v in parts.items():
        print(f"  {k:<28} {v:>7,}")
    print(f"  {'total input':<28} {used:>7,}  ({used / window:.1%} of the window)")
    print(f"  output reserve {reserve:,}; room left {window - reserve - used:,}")
    B = ContextBudget(window=8_000, reserve_output=1_000)
    B.add("system", "word " * 400)
    doc = B.add("docs", "word " * 3_000, cap=2_000)
    print(f"  small window demo: 8,000 window - 1,000 reserve; system {B.used['system']}, docs capped to "
          f"{B.used['docs']} ({len(doc.split())} words); remaining {B.remaining()}")

    print("\n=== 2. Ordering retrieved documents ===")
    query = "refund window for damaged items"
    docs = ["refund window is 30 days for damaged items", "damaged items ship back free",
            "shipping takes 3 to 7 days", "refund goes to the original card",
            "passwords need 12 characters", "warranty covers manufacturing defects"]
    scores = score_relevance(query, docs)
    for d, s in sorted(zip(docs, scores), key=lambda p: -p[1]):
        print(f"  {s:.2f}  {d}")
    print("  edges-first order (what the prompt sees, top to bottom):")
    for d in edges_first(docs, scores):
        print(f"    {scores[docs.index(d)]:.2f}  {d}")

    print("\n=== 3. History compaction ===")
    turns = [f"turn {i}: " + "detail " * 60 for i in range(14)]
    before = sum(count_tokens(t) for t in turns)
    after_turns = compact_history(turns, max_tokens=500)
    after = sum(count_tokens(t) for t in after_turns)
    print(f"  14 turns, {before:,} tokens -> {len(after_turns)} messages, {after:,} tokens")

    print("\n=== 4. Dynamic tool selection ===")
    q = "search the git log and read the file that changed"
    chosen = select_tools(q)
    print(f"  query: {q!r}")
    print(f"  all tools {len(TOOLS)} x {TOOL_TOKENS} = {len(TOOLS) * TOOL_TOKENS:,} tokens; "
          f"selected {len(chosen)} -> {len(chosen) * TOOL_TOKENS:,} tokens "
          f"({1 - len(chosen) / len(TOOLS):.0%} saved)")

    print("\n=== 5. Twenty calls, naive vs engineered ===")
    T = 20
    for lean, name in [(False, "naive (30 tools, 8 docs, full history)"), (True, "engineered (10 tools, 3 docs, compacted)")]:
        per = [input_tokens(t, lean) for t in range(1, T + 1)]
        print(f"  {name:<42} call 1 {per[0]:>6,}  call 20 {per[-1]:>6,}  total {sum(per):>8,}")
    tot = [sum(input_tokens(t, lean) for t in range(1, T + 1)) for lean in (False, True)]
    print(f"  engineered / naive = {tot[1] / tot[0]:.2f}")

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    left = 0
    colors = [PURPLE, BLUE, TEAL, ORANGE, GRAY]
    for (k, v), c in zip(parts.items(), colors):
        ax1.barh(0, v, left=left, color=c, label=f"{k}: {v:,}")
        left += v
    ax1.barh(0, reserve, left=window - reserve, color="#fde68a", label=f"output reserve: {reserve:,}")
    ax1.text(left + 2_000, 0, f"input used: {left:,} ({left / window:.1%})\nthe rest is empty", va="center", fontsize=9)
    ax1.set_xlim(0, window)
    ax1.set_yticks([])
    ax1.set_xlabel("tokens in a 128,000-token window")
    ax1.set_title("Where the tokens go (synthetic coding assistant)")
    ax1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ts = np.arange(1, 31)
    ax2.plot(ts, [input_tokens(t, False) for t in ts], "o-", ms=3, color=ORANGE, label="naive: everything, full history")
    ax2.plot(ts, [input_tokens(t, True) for t in ts], "o-", ms=3, color=TEAL, label="engineered: selected + compacted")
    ax2.set_xlabel("conversation turn (model call)")
    ax2.set_ylabel("input tokens for that call")
    ax2.set_ylim(0, None)
    ax2.set_title("Per-call input: growing vs bounded")
    ax2.legend(fontsize=8)
    save(fig, "context-engineering")

if __name__ == "__main__":
    main()
