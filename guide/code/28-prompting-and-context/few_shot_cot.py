# Adapted from AI Engineering from Scratch (MIT), phases/11-llm-engineering/02-few-shot-cot
"""Few-shot, chain-of-thought, and self-consistency on synthetic multi-step word problems.

ToyLLM is a scripted stand-in, not a real LLM. Its error rates are ASSUMPTIONS
chosen to illustrate the mechanism (answering directly slips 15% per step,
writing each step out slips 5% per step); they are not measurements of any model.
All problems are SYNTHETIC. Swap call_model() for a real client to test a real model.

Run from guide/:  uv run code/28-prompting-and-context/few_shot_cot.py
"""
import hashlib
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402

EPS_DIRECT, EPS_COT = 0.15, 0.05          # assumed per-step slip rates (synthetic)
OPS = {"buys {k} more": lambda x, k: x + k, "gives away {k}": lambda x, k: x - k,
       "doubles them": lambda x, k: 2 * x}
FORMATS = ["The answer is {a}.", "{a} stickers", "Answer: {a}", "So she ends up with {a}"]


# ---------- synthetic problems ----------
def make_problem(rng, steps):
    x = int(rng.integers(5, 20))
    text, answer = [f"Mia has {x} stickers."], x
    for _ in range(steps):
        op = list(OPS)[rng.integers(0, 3)]
        k = int(rng.integers(2, 9))
        if op.startswith("gives") and k >= answer:      # keep counts positive
            op = "buys {k} more"
        answer = OPS[op](answer, k)
        text.append("She " + op.format(k=k) + ".")
    return " ".join(text) + " How many stickers does she have?", answer


# ---------- prompts ----------
EXAMPLE = ("Q: Mia has 7 stickers. She buys 3 more. She doubles them. How many stickers does she have?\n"
           "A: Start with 7. Buys 3 more: 7 + 3 = 10. Doubles them: 10 * 2 = 20.\nAnswer: 20")


def zero_shot(q):
    return f"Q: {q}\nA:"


def zero_shot_cot(q):
    return f"Q: {q}\nA: Let's think step by step."


def few_shot_cot(q):
    return f"{EXAMPLE}\n\nQ: {q}\nA:"


# ---------- ToyLLM: reads the prompt, imitates example formats, slips at a per-step rate ----------
def toy_generate(prompt, temperature=0.0, seed=0):
    q = prompt.rsplit("Q: ", 1)[1].split("\nA:")[0]
    x = int(re.search(r"has (\d+)", q).group(1))
    steps = re.findall(r"She (buys (\d+) more|gives away (\d+)|doubles them)\.", q)
    cot = "step by step" in prompt or "Start with" in prompt          # reasoning requested or shown
    eps = EPS_COT if cot else EPS_DIRECT
    key = f"{q}|{seed if temperature > 0 else 0}"                     # greedy: same answer every call
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    lines = [f"Start with {x}."]
    for full, buy, give in steps:
        new = x + int(buy) if buy else x - int(give) if give else 2 * x
        if rng.random() < eps:                                         # a slip: off by 1 to 3
            new += int(rng.choice([-3, -2, -1, 1, 2, 3]))
        lines.append(f"{full}: {new}.")
        x = new
    if "Answer:" in prompt:                                            # copies the example format
        tail = f"Answer: {x}"
    else:
        tail = FORMATS[rng.integers(0, len(FORMATS))].format(a=x)
    return (" ".join(lines) + "\n" + tail) if cot else tail


def call_model(prompt, temperature=0.0, seed=0):
    """Swap for a real client, for example Anthropic messages.create(...).content[0].text."""
    return toy_generate(prompt, temperature, seed)


# ---------- answer extraction and voting ----------
def extract_strict(text):
    m = re.search(r"Answer:\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


def extract_last_number(text):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def self_consistency(q, n=5, temperature=0.7):
    answers = [extract_strict(call_model(few_shot_cot(q), temperature, seed)) for seed in range(n)]
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None, answers


def p_majority(p, n):
    """P(more than half of n independent chains are right): a lower bound on voting accuracy."""
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


def main():
    rng = np.random.default_rng(0)
    q, a = make_problem(rng, 3)
    print(f"Synthetic problem: {q}\nTrue answer: {a}\n")
    for name, f in [("zero-shot", zero_shot), ("zero-shot CoT", zero_shot_cot), ("few-shot CoT", few_shot_cot)]:
        out = call_model(f(q))
        print(f"--- {name} ---\n{out}\n  strict extract: {extract_strict(out)}   last number: {extract_last_number(out)}")

    probs = [make_problem(np.random.default_rng(1000 + i), 4) for i in range(300)]
    strict0 = np.mean([extract_strict(call_model(zero_shot_cot(p))) is not None for p, _ in probs])
    strict1 = np.mean([extract_strict(call_model(few_shot_cot(p))) is not None for p, _ in probs])
    print(f"\nFormat: strict 'Answer: N' found in {strict0:.0%} of zero-shot CoT outputs, {strict1:.0%} of few-shot CoT.")

    steps_range = range(1, 9)
    acc = {"direct": [], "cot": [], "sc5": []}
    for s in steps_range:
        ps = [make_problem(np.random.default_rng(10_000 * s + i), s) for i in range(300)]
        acc["direct"].append(np.mean([extract_last_number(call_model(zero_shot(p))) == y for p, y in ps]))
        acc["cot"].append(np.mean([extract_strict(call_model(few_shot_cot(p))) == y for p, y in ps]))
        acc["sc5"].append(np.mean([self_consistency(p)[0] == y for p, y in ps]))
    print("\nAccuracy on 300 synthetic problems per step count (last-number extraction for direct):")
    print(f"{'steps':>5} {'direct':>8} {'(0.85^s)':>9} {'CoT':>7} {'(0.95^s)':>9} {'SC N=5':>8} {'(majority bound)':>17}")
    for i, s in enumerate(steps_range):
        pc = (1 - EPS_COT) ** s
        print(f"{s:>5} {acc['direct'][i]:>8.3f} {(1 - EPS_DIRECT) ** s:>9.3f} {acc['cot'][i]:>7.3f} "
              f"{pc:>9.3f} {acc['sc5'][i]:>8.3f} {p_majority(pc, 5):>17.3f}")

    print(f"\nWorked example: p = 0.7 per chain, N = 5 -> P(at least 3 right) = {p_majority(0.7, 5):.5f}")
    print(f"Systematic error (wrong chains agree): p = 0.4, N = 5 -> {p_majority(0.4, 5):.5f}, worse than one chain")
    q4, y4 = probs[0]
    best, answers = self_consistency(q4)
    print(f"\nOne self-consistency run: answers {answers} -> vote {best} (true {y4})")

    # ---------- figure ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    s = list(steps_range)
    ax.plot(s, acc["direct"], "o-", color=ORANGE, label="answer directly")
    ax.plot(s, acc["cot"], "o-", color=BLUE, label="few-shot chain-of-thought")
    ax.plot(s, acc["sc5"], "o-", color=TEAL, label="self-consistency (5 chains, vote)")
    ax.plot(s, [(1 - EPS_DIRECT) ** k for k in s], ":", color=GRAY, lw=1)
    ax.plot(s, [(1 - EPS_COT) ** k for k in s], ":", color=GRAY, lw=1, label="analytic (1 - ε)^s")
    ax.set_xlabel("reasoning steps in the problem (s)")
    ax.set_ylabel("accuracy (300 synthetic problems per point)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Errors compound per step; writing steps and voting slow the decay")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "few-shot-cot")


if __name__ == "__main__":
    main()
