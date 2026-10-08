---
title: The end-to-end ML workflow
summary: Frame, collect, explore, split, baseline, train, evaluate, deploy, monitor, without leaking the answer.
skill: classical-ml
---

# Module 2: The end-to-end machine learning workflow

A model is one step of a loop. Most production failures come from the other steps: a split that doesn't match deployment, a feature that leaks the future, a metric that ignores the cost of errors, or no monitoring after launch.

| Step | Key question | Deliverable |
|---|---|---|
| 1. Frame | What decision and metric matter? | problem statement |
| 2. Collect | Is the data representative and permitted? | versioned dataset |
| 3. Explore | What is missing, skewed, duplicated, or leaked? | EDA report |
| 4. Split | How will future data differ? | train / validation / test sets |
| 5. Baseline | What does a simple rule achieve? | reference score |
| 6. Train | Which pipeline and model? | reproducible experiment |
| 7. Evaluate | Where and why does it fail? | error slices |
| 8. Deploy | How is inference served? | API or batch job |
| 9. Monitor | Has data or quality drifted? | alerts and dashboards |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Workflow, splits, and leakage](01-ml-workflow.md) | choose random, time, or group splits and find leakage |
| 2 | [Overfitting and the bias-variance trade-off](02-overfitting-and-bias-variance.md) | diagnose under- and overfitting from training and validation curves |
