# Adapted from AI Engineering from Scratch (MIT), phases/11-llm-engineering/11-caching-cost,
# phases/11-llm-engineering/15-prompt-caching, phases/13-tools-and-protocols/20-opentelemetry-genai,
# phases/17-infrastructure-and-production/14-prompt-semantic-caching, phases/17-infrastructure-and-production/16-model-routing
"""Exact cache, semantic cache, prompt-cache cost arithmetic, a model router, and GenAI trace spans.

All prices are SYNTHETIC (USD per million tokens); the multipliers 1.25x write / 0.1x read mirror the
shape of published prompt-caching price sheets but are not any provider's quote.
No model is called. Deterministic. Stdlib only.
Run from guide/:  uv run code/20-production-ai/cache_router_scratch.py
"""
import hashlib
import math
import re
from collections import Counter


# --- 1. Exact cache: key on everything that changes the answer -----------------------------
def cache_key(model, prompt_version, prompt, temperature):
    raw = f"{model}|{prompt_version}|{temperature}|{prompt.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --- 2. Semantic cache: bag-of-words cosine against stored questions ---------------------------
def bow(text):
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(self, threshold):
        self.threshold, self.items = threshold, []      # ponytail: linear scan, use a vector index past ~10k entries

    def get(self, q):
        best = max(self.items, key=lambda it: cosine(bow(q), it[0]), default=None)
        if best and cosine(bow(q), best[0]) >= self.threshold:
            return best[1]
        return None

    def put(self, q, answer):
        self.items.append((bow(q), answer))


# --- 3. Prompt (prefix) caching cost -------------------------------------------------------
PRICE_IN, PRICE_OUT = 3.00, 15.00             # SYNTHETIC USD per 1M tokens
WRITE_MULT, READ_MULT = 1.25, 0.10            # cache write premium, cache read discount


def request_cost(prefix_tok, suffix_tok, out_tok, cache_state):
    """cache_state: 'none' (no caching), 'write' (first call stores prefix), 'read' (prefix hit)."""
    mult = {"none": 1.0, "write": WRITE_MULT, "read": READ_MULT}[cache_state]
    return (prefix_tok * mult * PRICE_IN + suffix_tok * PRICE_IN + out_tok * PRICE_OUT) / 1e6


# --- 4. Router: cheap model unless the request looks hard ----------------------------------
HARD_WORDS = {"prove", "derive", "refactor", "debug", "legal", "contract", "multi-step", "why"}


def route(prompt):
    words = set(re.findall(r"[a-z\-]+", prompt.lower()))
    score = len(words & HARD_WORDS) + (1 if len(prompt) > 400 else 0)
    return ("large-model" if score >= 1 else "small-model"), score


# --- 5. One trace span per LLM call, GenAI semantic-convention attribute names -------------
def chat_span(model, provider, in_tok, out_tok, latency_ms, cache_read_tok=0):
    return {
        "name": f"chat {model}",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": in_tok,
            "gen_ai.usage.output_tokens": out_tok,
            "app.cache_read_tokens": cache_read_tok,      # app-specific attrs: own namespace, not gen_ai.*
            "app.latency_ms": latency_ms,
        },
    }


if __name__ == "__main__":
    k1 = cache_key("m-1", "faq-v3", "How do I reset my password?", 0.0)
    k2 = cache_key("m-1", "faq-v3", "  how do I reset my password?", 0.0)
    k3 = cache_key("m-1", "faq-v4", "How do I reset my password?", 0.0)
    print("Exact cache: same key after normalising?", k1 == k2, "| new prompt version -> new key?", k1 != k3)
    assert k1 == k2 and k1 != k3

    print("Semantic cache (stored: 'how do i reset my password'):")
    probes = ["how can i reset my password", "how do i reset my username", "what is your refund policy"]
    for th in (0.8, 0.9):
        c = SemanticCache(th)
        c.put("how do i reset my password", "Settings > Security > Reset password.")
        hits = [(p, round(cosine(bow(p), bow("how do i reset my password")), 2), c.get(p) is not None) for p in probes]
        print(f"  threshold {th}:", hits)
    # The paraphrase and the different-intent question score the SAME (0.83): no threshold separates them.
    # At 0.8 the 'username' question is a FALSE HIT; at 0.9 the paraphrase is a miss.
    c8 = SemanticCache(0.8); c8.put("how do i reset my password", "a")
    c9 = SemanticCache(0.9); c9.put("how do i reset my password", "a")
    assert c8.get("how do i reset my username") is not None      # false hit
    assert c9.get("how can i reset my password") is None          # missed paraphrase

    print("Prompt caching, 4000-token shared prefix, 200 new tokens in, 300 out, 100 calls:")
    no_cache = 100 * request_cost(4000, 200, 300, "none")
    cached = request_cost(4000, 200, 300, "write") + 99 * request_cost(4000, 200, 300, "read")
    print(f"  no cache USD {no_cache:.4f}   cached USD {cached:.4f}   saving {1 - cached / no_cache:.0%}")
    assert abs(request_cost(4000, 200, 300, "none") - 0.0171) < 1e-9
    assert abs(request_cost(4000, 200, 300, "read") - 0.0063) < 1e-9

    print("Router:")
    for p in ["What are your opening hours?", "Why does this recursive function overflow? Debug it.",
              "Summarize this contract clause for me."]:
        print(f"  {route(p)[0]:12} score={route(p)[1]}  {p}")
    assert route("What are your opening hours?")[0] == "small-model"

    span = chat_span("small-model", "example", 4200, 300, 410, cache_read_tok=4000)
    print("Span:", span["name"], {k: v for k, v in span["attributes"].items() if k.startswith("gen_ai.")})
    print("all checks passed")
