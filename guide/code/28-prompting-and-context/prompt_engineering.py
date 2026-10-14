# Adapted from AI Engineering from Scratch (MIT), phases/11-llm-engineering/01-prompt-engineering
"""Prompt engineering as an engineering loop: template -> model -> scorer -> pass rates.

The "model" is ToyLLM, a scripted stand-in that follows instructions literally:
it only uses the label set, output format, and edge-case rules the prompt spells
out, and guesses otherwise. It is NOT a real LLM and its pass rates are not
benchmarks; it exists so the harness runs offline and deterministically.
The 24 reviews are SYNTHETIC. To use a real model, replace call_model() (see
the commented client code below).

Run from guide/:  uv run code/28-prompting-and-context/prompt_engineering.py
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402

# ---------- synthetic test set: (review, gold label) ----------
REVIEWS = [
    ("Great battery and a lovely screen.", "positive"),
    ("Fast shipping, works perfectly.", "positive"),
    ("Love it, excellent value.", "positive"),
    ("Sturdy build and great sound.", "positive"),
    ("Excellent support team, very helpful.", "positive"),
    ("Perfect fit, love the color.", "positive"),
    ("Not bad at all for the price.", "positive"),
    ("Lovely design and fast setup.", "positive"),
    ("Broken on arrival, terrible.", "negative"),
    ("Slow, buggy, and the battery died.", "negative"),
    ("Awful support, never again.", "negative"),
    ("Cheap plastic that cracked in a week.", "negative"),
    ("Terrible sound and a broken hinge.", "negative"),
    ("Refund took forever, awful experience.", "negative"),
    ("It stopped charging after two days.", "negative"),
    ("Never worked, total waste.", "negative"),
    ("Great screen but the battery died fast.", "mixed"),
    ("Terrible packaging, but the product works great.", "mixed"),
    ("Love the sound, hate the slow app.", "mixed"),
    ("Excellent camera, awful battery.", "mixed"),
    ("Broken hinge, though support was helpful.", "mixed"),
    ("Fast setup but buggy software.", "mixed"),
    ("Lovely color, cheap plastic.", "mixed"),
    ("Perfect size, terrible speaker.", "mixed"),
]
POS = {"great", "lovely", "fast", "perfectly", "love", "excellent", "sturdy", "helpful", "perfect", "works"}
NEG = {"broken", "terrible", "slow", "buggy", "died", "awful", "never", "cheap", "cracked", "forever", "waste", "hate", "bad"}
LABELS = ["positive", "negative", "mixed"]


# ---------- prompt templates (a template is a function of variables) ----------
PROMPTS = {
    "v0 vague": "What do you think of this review?\n\n{review}",
    "v1 +labels": "Classify the sentiment of the review as one of: positive, negative, mixed.\n\n{review}",
    "v2 +JSON": ("Classify the sentiment of the review as one of: positive, negative, mixed.\n"
                 'Respond as JSON: {{"sentiment": "<label>"}}.\n\n<review>{review}</review>'),
    "v3 +rules": ("Classify the sentiment of the review as one of: positive, negative, mixed.\n"
                  "Use mixed when the review praises one thing and criticizes another.\n"
                  'Respond as JSON: {{"sentiment": "<label>"}}. Output only the JSON, no other text.\n\n'
                  "<review>{review}</review>"),
}
SYSTEM = "You are a careful analyst of customer reviews."


def build_prompt(template, **variables):
    missing = [k for k in ("review",) if k not in variables]
    if missing:
        raise ValueError(f"missing variables: {missing}")
    return template.format(**variables)


# ---------- ToyLLM: follows instructions literally, guesses where the prompt is silent ----------
class ToyLLM:
    def generate(self, system, user, temperature=0.0, seed=0):
        review = user.split("<review>")[-1].split("</review>")[0].split("\n\n")[-1]
        words = [w.strip(".,!").lower() for w in review.split()]
        pos, neg = sum(w in POS for w in words), sum(w in NEG for w in words)
        knows_mixed_rule = "praises one thing" in user          # edge case defined?
        scores = {"positive": pos - neg, "negative": neg - pos,
                  "mixed": (min(pos, neg) + 0.5) if knows_mixed_rule else (min(pos, neg) - 1.0)}
        label = self._pick(scores, temperature, user, review, seed)
        if "one of: positive, negative, mixed" not in user:      # no label set: free vocabulary
            label = {"positive": "good", "negative": "bad", "mixed": "it is mixed"}[label]
        if "JSON" not in user:                                   # no format: chatty prose
            return f"Overall the customer seems {label} about it."
        body = json.dumps({"sentiment": label})
        return body if "Output only the JSON" in user else f"Here is the analysis:\n{body}"

    @staticmethod
    def _pick(scores, temperature, user, review, seed):
        names = list(scores)
        z = np.array([scores[n] for n in names], dtype=float)
        if temperature == 0:
            return names[int(np.argmax(z))]                       # greedy: deterministic
        h = int(hashlib.md5(f"{user}|{review}|{seed}".encode()).hexdigest()[:8], 16)
        p = np.exp(z / temperature - np.max(z / temperature))
        p /= p.sum()                                              # softmax with temperature
        return names[np.random.default_rng(h).choice(len(names), p=p)]


TOY = ToyLLM()


def call_model(system, user, temperature=0.0, seed=0):
    """The one place a real client plugs in. Offline default: ToyLLM."""
    return TOY.generate(system, user, temperature, seed)
    # Real model (pip install anthropic; set ANTHROPIC_API_KEY):
    # import anthropic
    # client = anthropic.Anthropic()
    # msg = client.messages.create(model=MODEL_ID, max_tokens=100, system=system,
    #                              messages=[{"role": "user", "content": user}],
    #                              temperature=temperature)
    # return msg.content[0].text


# ---------- scorer: strict parse, plus a lenient label check so we can see accuracy separately ----------
def score(output, gold):
    try:                                       # strict: what a downstream json.loads sees
        label, parsed = json.loads(output)["sentiment"], True
    except (json.JSONDecodeError, KeyError, TypeError):
        parsed = False                         # lenient: first allowed label word anywhere
        found = re.findall(r"\b(positive|negative|mixed)\b", output.lower())
        label = found[0] if found else None
    return {"parses": parsed, "valid_label": label in LABELS, "correct": label == gold}


def evaluate(template, temperature=0.0, seed=0):
    rows = [score(call_model(SYSTEM, build_prompt(template, review=r), temperature, seed), g) for r, g in REVIEWS]
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def std_err(p, n):
    return math.sqrt(p * (1 - p) / n)


def main():
    n = len(REVIEWS)
    print(f"Synthetic test set: {n} reviews (8 positive, 8 negative, 8 mixed). ToyLLM, temperature 0.\n")
    print(f"{'prompt':<12} {'parses':>7} {'valid':>7} {'correct':>8}")
    results = {}
    for name, tpl in PROMPTS.items():
        results[name] = evaluate(tpl)
        r = results[name]
        print(f"{name:<12} {r['parses']:>7.2f} {r['valid_label']:>7.2f} {r['correct']:>8.2f}")

    print("\nSample outputs for the mixed review 'Great screen but the battery died fast.':")
    for name, tpl in PROMPTS.items():
        print(f"  {name:<12} -> {call_model(SYSTEM, build_prompt(tpl, review=REVIEWS[16][0]))!r}")

    p2, p3 = results["v1 +labels"]["correct"], results["v3 +rules"]["correct"]
    se2, se3 = std_err(p2, n), std_err(p3, n)
    se_diff = math.sqrt(se2 ** 2 + se3 ** 2)
    print(f"\nIs v3 really better than v1?  p1 = {p2:.3f}, p3 = {p3:.3f}, n = {n}")
    print(f"  SE(p1) = {se2:.3f}, SE(p3) = {se3:.3f}, SE(diff) = {se_diff:.3f}, z = {(p3 - p2) / se_diff:.2f}")

    temps = [0.0, 0.3, 0.7, 1.0, 1.5]
    runs = {t: [evaluate(PROMPTS["v3 +rules"], t, seed)["correct"] for seed in range(10)] for t in temps}
    print("\nv3 accuracy over 10 seeds by temperature (mean, min, max):")
    for t in temps:
        a = runs[t]
        print(f"  T = {t:<4} mean {np.mean(a):.3f}  min {min(a):.3f}  max {max(a):.3f}")

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    names = list(PROMPTS)
    x = np.arange(len(names))
    for i, (k, c, lab) in enumerate([("parses", BLUE, "output parses as JSON"),
                                     ("valid_label", TEAL, "label in allowed set"),
                                     ("correct", ORANGE, "label correct")]):
        ax1.bar(x + (i - 1) * 0.26, [results[nm][k] for nm in names], 0.26, color=c, label=lab)
    ax1.set_xticks(x, names)
    ax1.set_ylim(0, 1.4)
    ax1.set_ylabel(f"pass rate on {n} synthetic reviews")
    ax1.set_title("Pass rates by prompt revision")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    means = [np.mean(runs[t]) for t in temps]
    ax2.plot(temps, means, "o-", color=PURPLE, label="mean of 10 seeds")
    ax2.fill_between(temps, [min(runs[t]) for t in temps], [max(runs[t]) for t in temps],
                     color=PURPLE, alpha=0.15, label="min to max")
    ax2.axhline(means[0], color=GRAY, ls="--", lw=1)
    ax2.set_xlabel("temperature")
    ax2.set_ylabel("accuracy (prompt v3)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Same prompt, higher temperature: lower, noisier")
    ax2.legend(fontsize=8)
    save(fig, "prompt-engineering")


if __name__ == "__main__":
    main()
