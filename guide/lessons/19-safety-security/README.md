---
title: Safety and security
summary: "Threat-model AI systems: prompt injection, data exfiltration, tool abuse, and the controls that contain them."
skill: engineering
---

# Module 19: Safety and security

When a model can read untrusted text and call tools, every document it reads is potential input from an attacker. Security for AI systems is mostly classic security (least privilege, validation, allowlists, audit logs) applied with the assumption that the model can be talked into anything.

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Securing AI systems](01-ai-security.md) | threat-model an LLM application and choose controls for prompt injection, exfiltration, and unsafe actions |
