---
title: Security controls
summary: Threats to LLM applications and the enforceable controls that contain them.
---

# Security controls

**Principle:** assume the model can be manipulated; make sure a manipulated model can't do much harm.

| Threat | Control |
|---|---|
| Direct prompt injection | enforce policy in code; no secrets in prompts; output filtering |
| Indirect prompt injection (in documents, pages, emails, tool outputs) | label untrusted content; task-scoped tool allowlists; validate actions; human approval |
| Data exfiltration (links, images, encoded data) | restrict or allowlist outbound URLs and rendered images |
| Tool abuse / excessive permissions | per-user, least-privilege credentials; argument schemas; rate limits |
| Secret leakage | keep keys server-side; redact logs |
| Unsafe generated SQL or code | parse and allowlist; read-only roles; sandbox without network; resource limits |
| Privacy violations | permission filters at retrieval; data minimization; retention limits |
| Supply chain | trusted model sources, checksums, safe serialization, pinned dependencies |
| Missing accountability | audit logs with request, retrieved IDs, tool calls, and versions |

**Layered defense math:** harm per attempt $= p_{\text{inj}}\prod_i p_{\text{control}_i}$; removing a capability sets its term to zero.

**Test:** keep injection, exfiltration, and privilege-escalation cases in the evaluation set; gate releases on attack success rate; red-team regularly. Demonstrate attacks only with mocked tools.

Lessons: [Securing AI systems](../lessons/19-safety-security/01-ai-security.md) · [Production architecture](../lessons/18-production-ai/01-production-architecture.md)
