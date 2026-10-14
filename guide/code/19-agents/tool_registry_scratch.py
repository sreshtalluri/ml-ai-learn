# Adapted from AI Engineering from Scratch (MIT), phases/14-agent-engineering/01-the-agent-loop,
# phases/13-tools-and-protocols/01-the-tool-interface, phases/13-tools-and-protocols/02-function-calling-deep-dive
"""A tool-calling agent from scratch: tool registry, JSON-schema argument check, errors returned as
observations, and a step-limited loop. Stdlib only.

The "model" is a SCRIPTED policy (no API calls); flight data is SYNTHETIC. Deterministic.
Run from guide/:  uv run code/19-agents/tool_registry_scratch.py
"""
import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str          # the model only sees this text and the schema
    input_schema: dict        # JSON Schema for the arguments
    fn: Callable[..., dict]


REGISTRY: dict[str, Tool] = {}


def tool(description: str, input_schema: dict):
    """Decorator: register a Python function as a tool the model can call."""
    def wrap(fn):
        REGISTRY[fn.__name__] = Tool(fn.__name__, description, input_schema, fn)
        return fn
    return wrap


def validate(schema: dict, value) -> list[str]:
    """Check a value against a small JSON Schema subset (object, string, number, enum)."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"expected object, got {type(value).__name__}"]
        errors = [f"missing required field '{k}'" for k in schema.get("required", []) if k not in value]
        errors += [f"unknown field '{k}'" for k in value if k not in schema.get("properties", {})]
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errors += [f"{k}: {e}" for e in validate(sub, value[k])]
        return errors
    if t == "string" and not isinstance(value, str):
        return [f"expected string, got {type(value).__name__}"]
    if t == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return [f"expected number, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{value!r} not in {schema['enum']}"]
    return []


FLIGHTS = {  # SYNTHETIC data
    "UA412": {"price": 289, "refundable": False, "change_fee": 99},
    "AS330": {"price": 312, "refundable": True, "change_fee": 0},
}


@tool("Use when the user wants flights between two airports. Returns flight_id and price. "
      "Do not use for fare rules; call get_fare_rules with a flight_id instead.",
      {"type": "object", "properties": {"origin": {"type": "string"}, "dest": {"type": "string"}},
       "required": ["origin", "dest"]})
def search_flights(origin, dest):
    return {"flights": [{"flight_id": k, "price": v["price"]} for k, v in FLIGHTS.items()]}


@tool("Use when you have a flight_id from search_flights and need refundability or change fee. "
      "Do not use to search for flights.",
      {"type": "object", "properties": {"flight_id": {"type": "string"}}, "required": ["flight_id"]})
def get_fare_rules(flight_id):
    if flight_id not in FLIGHTS:
        raise KeyError(f"flight_id {flight_id} not found; call search_flights first")
    f = FLIGHTS[flight_id]
    return {"flight_id": flight_id, "refundable": f["refundable"], "change_fee": f["change_fee"]}


def dispatch(name: str, args: dict) -> dict:
    """Run one tool call. Every failure becomes an observation the model can read, never a crash."""
    t = REGISTRY.get(name)
    if t is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(REGISTRY)}"}
    if errs := validate(t.input_schema, args):
        return {"error": "invalid arguments: " + "; ".join(errs)}
    try:
        return t.fn(**args)
    except Exception as e:                      # tool bug or bad id: report, don't crash the loop
        return {"error": f"{type(e).__name__}: {e}"}


def to_anthropic(t: Tool) -> dict:
    """Provider wire format. OpenAI wraps the same fields as {"type": "function", "function": {..., "parameters": ...}}."""
    return {"name": t.name, "description": t.description, "input_schema": t.input_schema}


class ScriptedModel:
    """Stands in for a chat API. Each reply is {"tool_calls": [...]} or {"text": ...}."""
    def __init__(self, replies):
        self.replies, self.i = replies, 0

    def __call__(self, messages, tools):
        reply = self.replies[min(self.i, len(self.replies) - 1)]   # last reply repeats forever
        self.i += 1
        return reply


def run_agent(task: str, model, max_steps: int = 6):
    messages = [{"role": "user", "content": task}]
    tools = [to_anthropic(t) for t in REGISTRY.values()]
    for step in range(1, max_steps + 1):          # hard stop owned by the runtime
        reply = model(messages, tools)
        messages.append({"role": "assistant", "content": reply})
        if "tool_calls" not in reply:             # no tool call means final answer
            return reply["text"], step, messages
        for call in reply["tool_calls"]:          # parallel calls: run each, append each result
            result = dispatch(call["name"], call["args"])
            messages.append({"role": "tool", "name": call["name"], "content": result})
            print(f"  step {step}: {call['name']}({json.dumps(call['args'])}) -> {json.dumps(result)}")
    return "Stopped: step limit reached.", max_steps, messages


if __name__ == "__main__":
    print("Tool schema sent to the model (Anthropic shape):")
    print(" ", json.dumps(to_anthropic(REGISTRY["get_fare_rules"]))[:110], "...")

    print("\nRun 1: scripted model makes one bad call, reads the error, recovers")
    model = ScriptedModel([
        {"tool_calls": [{"name": "search_flights", "args": {"origin": "SEA", "dest": "SFO"}}]},
        {"tool_calls": [{"name": "get_fare_rules", "args": {"flight_id": 412}}]},          # wrong type
        {"tool_calls": [{"name": "get_fare_rules", "args": {"flight_id": "UA412"}},
                        {"name": "get_fare_rules", "args": {"flight_id": "AS330"}}]},     # parallel
        {"text": "UA412 is cheapest (289) but non-refundable; AS330 (312) is refundable."},
    ])
    answer, steps, msgs = run_agent("Cheapest refundable SEA->SFO?", model)
    print(f"  final ({steps} model calls): {answer}")
    assert steps == 4 and "AS330" in answer
    assert "expected string" in msgs[4]["content"]["error"]

    print("\nRun 2: a model stuck calling a tool that does not exist")
    stuck = ScriptedModel([{"tool_calls": [{"name": "book_flight", "args": {"flight_id": "AS330"}}]}])
    answer, steps, _ = run_agent("Book AS330", stuck, max_steps=3)
    print(f"  final ({steps} model calls): {answer}")
    assert answer.startswith("Stopped")
    print("\nself-checks passed")
