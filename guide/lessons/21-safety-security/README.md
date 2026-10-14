---
title: Safety and security
summary: "Threat-model AI systems (prompt injection, data exfiltration, tool abuse), measure and choose fairness criteria, and protect training data with differential privacy."
skill: engineering
---

# Module 21: Safety and security

When a model can read untrusted text and call tools, every document it reads is potential input from an attacker. Security for AI systems is mostly classic security (least privilege, validation, allowlists, audit logs) applied with the assumption that the model can be talked into anything.

Two more risks sit in the data rather than the prompt. A model can be accurate overall and still fail one group of people far more often, and a model trained on personal data can memorize and repeat it. Fairness criteria and differential privacy are the tools for measuring and bounding those harms.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Securing AI systems](01-ai-security.md) | threat-model an LLM application and choose controls for prompt injection, exfiltration, and unsafe actions |
| 2 | [Fairness](02-fairness.md) | compute per-group selection, error, and precision rates, explain why they conflict under unequal base rates, and choose and audit a criterion |
| 3 | [Differential privacy](03-differential-privacy.md) | apply the Laplace and Gaussian mechanisms, train with DP-SGD and track ε, and decide when DP is the right defense against memorization |
