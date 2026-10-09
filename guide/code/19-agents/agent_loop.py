"""A tiny tool-calling agent loop with a MOCKED model, plus the context-token arithmetic from the lesson.

The "model" is a scripted policy (no API calls), the tools return SYNTHETIC flight data, and prices per token
are SYNTHETIC. Everything is deterministic.
Run from guide/:  uv run code/19-agents/agent_loop.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, ORANGE, TEAL, plt, save  # noqa: E402

# ---------- tools: JSON schemas the model sees, plus the code the runtime runs ----------
TOOLS = [
    {"name": "search_flights",
     "description": "Search scheduled flights between two airports on one date. Returns flights sorted by price.",
     "input_schema": {"type": "object", "properties": {
         "origin": {"type": "string", "description": "IATA airport code, e.g. SFO"},
         "destination": {"type": "string", "description": "IATA airport code"},
         "date": {"type": "string", "description": "YYYY-MM-DD"},
         "max_price_usd": {"type": "number"}},
         "required": ["origin", "destination", "date"]}},
    {"name": "get_fare_rules",
     "description": "Return the fare rules for one flight: refundable (bool) and change fee.",
     "input_schema": {"type": "object", "properties": {"flight_id": {"type": "string"}}, "required": ["flight_id"]}},
    {"name": "hold_booking",
     "description": "Place a 24-hour hold on a seat. SIDE EFFECT: reserves inventory. Needs user approval.",
     "input_schema": {"type": "object", "properties": {"flight_id": {"type": "string"}, "passenger_id": {"type": "string"}},
                      "required": ["flight_id", "passenger_id"]}},
    {"name": "web_search",
     "description": "General web search. Not a source of live prices or fare rules.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]
SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}
SIDE_EFFECTS = {"hold_booking"}

FLIGHTS = [  # SYNTHETIC
    {"flight_id": "UA412", "price_usd": 189, "fare_class": "basic"},
    {"flight_id": "AS330", "price_usd": 214, "fare_class": "main"},
    {"flight_id": "DL1180", "price_usd": 262, "fare_class": "main"},
    {"flight_id": "AA95", "price_usd": 341, "fare_class": "main"},
]
RULES = {"UA412": {"refundable": False, "change_fee_usd": 99}, "AS330": {"refundable": True, "change_fee_usd": 0},
         "DL1180": {"refundable": True, "change_fee_usd": 0}, "AA95": {"refundable": True, "change_fee_usd": 0}}


class Runtime:
    def __init__(self, fail_first_search=False):
        self.fail_first_search = fail_first_search
        self.calls = 0

    def run(self, name, args):
        self.calls += 1
        if name == "search_flights":
            if self.fail_first_search:
                self.fail_first_search = False
                return {"error": "503 upstream timeout", "retryable": True}
            cap = args.get("max_price_usd", float("inf"))
            return {"flights": [f for f in FLIGHTS if f["price_usd"] <= cap]}
        if name == "get_fare_rules":
            return RULES.get(args["flight_id"], {"error": f"unknown flight_id {args['flight_id']}", "retryable": False})
        if name == "hold_booking":
            return {"hold_id": "H-7Q2", "flight_id": args["flight_id"], "expires_in_hours": 24}
        return {"results": ["(irrelevant web pages)"]}


def validate(name, args):
    """Minimal JSON-schema check: known tool, required keys, no unknown keys, primitive types."""
    if name not in SCHEMAS:
        return f"unknown tool {name}"
    schema = SCHEMAS[name]
    missing = [k for k in schema["required"] if k not in args]
    extra = [k for k in args if k not in schema["properties"]]
    types = {"string": str, "number": (int, float)}
    bad = [k for k, v in args.items() if k in schema["properties"] and not isinstance(v, types[schema["properties"][k]["type"]])]
    return f"missing={missing} extra={extra} bad_type={bad}" if missing or extra or bad else None


def mock_model(history, task):
    """Scripted stand-in for the LLM: reads the transcript, returns a tool call or a final answer."""
    last = history[-1]["content"] if history and history[-1]["role"] == "tool" else None
    if last and last.get("retryable"):
        prev = history[-2]["content"]
        return {"thought": "Transient error; retry the same call once.", "call": prev}
    flights = next((m["content"]["flights"] for m in history if m["role"] == "tool" and "flights" in m["content"]), None)
    if flights is None:
        return {"thought": "I need candidate flights first.",
                "call": {"name": "search_flights", "args": {"origin": "SFO", "destination": "SEA",
                                                            "date": "2026-11-14", "max_price_usd": task["budget"]}}}
    checked = {m["call"]["args"]["flight_id"]: m["content"] for m in history
               if m["role"] == "tool" and m["call"]["name"] == "get_fare_rules"}
    held = next((m["content"] for m in history if m["role"] == "tool" and m["call"]["name"] == "hold_booking"), None)
    if held:
        return {"thought": "Done.", "final": f"Held {held['flight_id']} (hold {held['hold_id']}, 24 h)."}
    for f in flights:  # cheapest first
        if f["flight_id"] not in checked:
            return {"thought": f"Is {f['flight_id']} refundable?",
                    "call": {"name": "get_fare_rules", "args": {"flight_id": f["flight_id"]}}}
        if checked[f["flight_id"]].get("refundable"):
            return {"thought": f"{f['flight_id']} is the cheapest refundable option; hold it.",
                    "call": {"name": "hold_booking", "args": {"flight_id": f["flight_id"], "passenger_id": "traveler-1"}}}
    return {"thought": "Nothing qualifies.", "final": "No refundable flight under budget."}


def agent_loop(task, runtime, max_steps=8, approve=lambda call: True):
    history, steps = [], 0
    while steps < max_steps:
        steps += 1
        action = mock_model(history, task)                   # 1. model call (reads the whole history)
        print(f"  step {steps}: {action['thought']}")
        if "final" in action:                                 # 2. stop condition: final answer
            print(f"    final: {action['final']}")
            return action["final"], steps
        call = action["call"]
        if (err := validate(call["name"], call["args"])):     # 3. validate against the schema
            result = {"error": f"invalid arguments: {err}", "retryable": False}
        elif call["name"] in SIDE_EFFECTS and not approve(call):
            return "User declined the hold.", steps           # 4. human approval gate for side effects
        else:
            result = runtime.run(call["name"], call["args"])  # 5. execute and append the observation
        print(f"    call: {json.dumps(call)}\n    obs:  {json.dumps(result)}")
        history.append({"role": "assistant", "content": call})
        history.append({"role": "tool", "call": call, "content": result})
    return "Stopped: step limit reached.", steps


def cumulative_input_tokens(base, per_step, steps):
    """Every model call re-reads the whole context: sum_{t=1}^{T} (base + (t-1)*per_step)."""
    return steps * base + per_step * steps * (steps - 1) // 2


if __name__ == "__main__":
    task = {"goal": "Cheapest refundable SFO->SEA flight on 2026-11-14 under budget; hold it.", "budget": 300}
    print("Happy path:")
    agent_loop(task, Runtime())
    print("\nWith a transient tool error (agent retries):")
    agent_loop(task, Runtime(fail_first_search=True))
    print("\nWith max_steps=3 (agent stops before holding):")
    print("  ->", agent_loop(task, Runtime(), max_steps=3)[0])
    print("\nInvalid call caught by the schema check:", validate("get_fare_rules", {"flight": "UA412"}))

    # ---------- worked example: token growth ----------
    system, schemas, user = 500, 600, 100
    base = system + schemas + user
    out_tok, obs_tok = 80, 320
    per = out_tok + obs_tok
    T = 6
    ctx = [base + (t - 1) * per for t in range(1, T + 1)]
    total_in = cumulative_input_tokens(base, per, T)
    total_out = T * out_tok
    print(f"\nBase context = {system} + {schemas} + {user} = {base} tokens; each step adds {out_tok} + {obs_tok} = {per}")
    print("Input tokens per call:", ctx, "sum =", sum(ctx))
    assert total_in == sum(ctx) == 13_200
    p_in, p_out, p_cached = 3.0, 15.0, 0.30   # SYNTHETIC USD per million tokens
    cost = total_in * p_in / 1e6 + total_out * p_out / 1e6
    fresh = base + (T - 1) * per
    cached = total_in - fresh
    cost_cached = fresh * p_in / 1e6 + cached * p_cached / 1e6 + total_out * p_out / 1e6
    print(f"Output tokens = {T} x {out_tok} = {total_out}")
    print(f"Cost, no caching   = {total_in}*3e-6 + {total_out}*15e-6 = {cost:.4f} USD")
    print(f"With prefix cache: fresh = {fresh}, cached = {cached}; cost = {cost_cached:.4f} USD")

    # ---------- figure ----------
    Ts = list(range(1, 11))
    per_call = [base + (t - 1) * per for t in Ts]
    cum = [cumulative_input_tokens(base, per, t) for t in Ts]
    cost_nc = [cumulative_input_tokens(base, per, t) * p_in / 1e6 + t * out_tok * p_out / 1e6 for t in Ts]
    cost_c = [((base + (t - 1) * per) * p_in + (cumulative_input_tokens(base, per, t) - base - (t - 1) * per) * p_cached
               + t * out_tok * p_out) / 1e6 for t in Ts]
    fig, (a, c, b) = plt.subplots(1, 3, figsize=(13, 3.8))
    a.bar(Ts, per_call, color=BLUE, alpha=0.85)
    a.set(xlabel="agent step (model call)", ylabel="input tokens", title="Each call re-reads the context")
    c.plot(Ts, cum, color=ORANGE, marker="o")
    c.set(xlabel="agent step", ylabel="cumulative input tokens", title="Total grows quadratically")
    c.annotate(f"6 steps: {cum[5]:,}", (6, cum[5]), xytext=(1.5, cum[-1] * 0.8), arrowprops={"arrowstyle": "->"})
    b.plot(Ts, cost_nc, color=ORANGE, marker="o", label="no caching")
    b.plot(Ts, cost_c, color=TEAL, marker="s", label="prefix caching (cached input at 10%)")
    b.set(xlabel="agent step", ylabel="cumulative cost (USD, synthetic prices)", title="Prefix caching flattens the curve")
    b.legend(fontsize=8)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "tool-use-and-agents")
