---
title: ML system design and experimentation
summary: "Design an ML system end to end with a repeatable framework, size a recommendation funnel against a latency budget, and prove it works with a well-powered A/B test."
skill: engineering
---

# Module 22: ML system design and experimentation

A model is one box in a system. This module covers the rest: framing the problem, choosing metrics, building labels and features without leakage, serving within a latency budget, and deciding with an online experiment whether the new system is actually better.

| Step | What you decide | Failure mode |
|---|---|---|
| Clarify and frame | goal, constraints, prediction target | optimizing the wrong thing |
| Metrics | business, online, offline, guardrails | offline wins that lose online |
| Data and features | label source, point-in-time features, feature store | leakage, training-serving skew |
| Model and serving | baseline, funnel stages, latency budget | blowing the budget, no fallback |
| Experiment | sample size, duration, randomization unit | peeking, SRM, interference |
| Monitor | drift, calibration, freshness | silent decay, feedback loops |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [ML system design](01-ml-system-design.md) | run a ten-step design framework and design a multi-stage recommendation funnel with a latency budget |
| 2 | [Experimentation and A/B testing](02-experimentation-and-ab-testing.md) | compute sample size and power, read a z-test, and catch peeking, SRM, novelty, and interference |
