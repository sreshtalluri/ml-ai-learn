# Adapted from AI Engineering from Scratch (MIT), phases/14-agent-engineering/12-anthropic-workflow-patterns,
# phases/14-agent-engineering/05-self-refine-and-critic, phases/14-agent-engineering/26-failure-modes-agentic,
# phases/15-autonomous-systems/13-cost-governors, phases/15-autonomous-systems/15-propose-then-commit
"""Four runtime guards that make agents reliable, built from scratch with the stdlib:
evaluator-optimizer with a grounded checker, a repeated-call detector, a cost governor,
and propose-then-commit for side-effecting tools.

Every "model" is SCRIPTED (no API calls). Token counts and prices are SYNTHETIC. Deterministic.
Run from guide/:  uv run code/19-agents/reliability_scratch.py
"""
import hashlib
import json


# 1. Evaluator-optimizer: generate, check against ground truth, feed the critique back.
def evaluator_optimizer(propose, evaluate, max_iter=4):
    feedback, trace = None, []
    for _ in range(max_iter):
        draft = propose(feedback)
        ok, feedback = evaluate(draft)
        trace.append((draft, ok, feedback))
        if ok:
            break
    return draft, trace


OBSERVED = {"AS330": {"refundable": True}, "UA412": {"refundable": False}}  # tool results (SYNTHETIC)


def grounded_check(draft: str):
    """CRITIC-style check: every refundability claim must match an observation, not the model's own opinion."""
    for fid, obs in OBSERVED.items():
        if fid in draft:
            claims_refund = f"{fid} is refundable" in draft
            if claims_refund != obs["refundable"]:
                return False, f"claim about {fid} contradicts tool result refundable={obs['refundable']}"
    return True, "ok"


# 2. Loop detector: the same call with the same arguments, again, means no progress.
class LoopDetector:
    def __init__(self, max_repeats=2):
        self.seen, self.max_repeats = {}, max_repeats

    def check(self, name: str, args: dict) -> bool:
        key = name + json.dumps(args, sort_keys=True)       # canonical form: key order must not matter
        self.seen[key] = self.seen.get(key, 0) + 1
        return self.seen[key] > self.max_repeats            # True means stop


# 3. Cost governor: charge every model call; halt before the budget is crossed, not after.
PRICE_IN, PRICE_OUT = 3e-6, 15e-6                            # USD per token (SYNTHETIC)


class BudgetExceeded(Exception):
    pass


class CostGovernor:
    def __init__(self, max_usd: float, max_steps: int):
        self.max_usd, self.max_steps, self.usd, self.steps = max_usd, max_steps, 0.0, 0

    def charge(self, tokens_in: int, tokens_out: int):
        cost = tokens_in * PRICE_IN + tokens_out * PRICE_OUT
        if self.steps + 1 > self.max_steps:
            raise BudgetExceeded(f"step cap {self.max_steps}")
        if self.usd + cost > self.max_usd:
            raise BudgetExceeded(f"next call costs {cost:.4f}, spent {self.usd:.4f} of {self.max_usd}")
        self.usd, self.steps = self.usd + cost, self.steps + 1


# 4. Propose-then-commit: a side effect runs only after approval, at most once per idempotency key.
BACKEND: dict[str, dict] = {}                                # pretend booking system


def idempotency_key(task_id: str, action: str, payload: dict) -> str:
    sig = json.dumps({"t": task_id, "a": action, "p": payload}, sort_keys=True)
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


def commit(task_id: str, action: str, payload: dict, approve) -> dict:
    key = idempotency_key(task_id, action, payload)
    if key in BACKEND:                                       # retry after a timeout: no double booking
        return {"status": "already_committed", "key": key}
    if not approve(action, payload):                         # human sees the evidence, not raw JSON
        return {"status": "declined", "key": key}
    BACKEND[key] = {"action": action, **payload}
    verified = BACKEND.get(key) == {"action": action, **payload}   # re-read the target after writing
    return {"status": "committed" if verified else "commit_unverified", "key": key}


if __name__ == "__main__":
    print("1. Evaluator-optimizer with a grounded checker")
    drafts = iter(["Book UA412: UA412 is refundable and cheapest.",       # false success claim
                   "UA412 is cheapest but not refundable; AS330 is refundable at 312."])
    final, trace = evaluator_optimizer(lambda fb: next(drafts), grounded_check)
    for i, (d, ok, fb) in enumerate(trace, 1):
        print(f"  iter {i} {'PASS' if ok else 'FAIL'}: {fb}")
    assert len(trace) == 2 and trace[-1][1]

    print("\n2. Loop detector on a stuck agent")
    det, calls = LoopDetector(max_repeats=2), [("search_flights", {"origin": "SEA", "dest": "SFO"})] * 5
    for step, (name, args) in enumerate(calls, 1):
        if det.check(name, args):
            print(f"  stopped at step {step}: {name} repeated {det.max_repeats + 1} times")
            break
    assert step == 3

    print("\n3. Cost governor on a run whose context grows 400 tokens per step")
    gov = CostGovernor(max_usd=0.05, max_steps=20)
    try:
        for t in range(1, 21):
            gov.charge(tokens_in=1_200 + 400 * (t - 1), tokens_out=80)
    except BudgetExceeded as e:
        print(f"  halted before call {t}: {e}")
    print(f"  calls made {gov.steps}, spent {gov.usd:.4f} USD")
    assert gov.steps == 6 and gov.usd <= 0.05

    print("\n4. Propose-then-commit with an idempotency key")
    hold = ("task-7", "hold_booking", {"flight_id": "AS330", "price": 312})
    print(" ", commit(*hold, approve=lambda a, p: False))    # human says no
    print(" ", commit(*hold, approve=lambda a, p: True))     # human says yes
    print(" ", commit(*hold, approve=lambda a, p: True))     # network retry
    assert len(BACKEND) == 1
    print("\nself-checks passed")
