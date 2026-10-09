---
title: ML system design
summary: The ten-step design framework as a checklist, the recommendation funnel, data pitfalls, and the A/B test formulas.
---

# ML system design

## Framework checklist (45 minutes)

- [ ] **1. Clarify (0 to 5 min).** Business goal, users and items, QPS, latency budget (p99), freshness, privacy and fairness constraints. State assumptions out loud.
- [ ] **2. Frame.** What is predicted, for whom, when. Classification, regression, ranking, or retrieval.
- [ ] **3. Metrics.** Business (retention, revenue) · online (CTR, watch time) · offline (AUC, log loss, recall@k, nDCG) · guardrails (latency, errors, diversity, safety).
- [ ] **4. Data and labels.** Source, delay, noise, selection and position bias, class balance.
- [ ] **5. Features.** User, item, context, cross. Batch versus real-time. One definition for training and serving (feature store).
- [ ] **6. Model.** Baseline first (popularity, rules, logistic regression), then GBDT, then deep models if they earn their cost.
- [ ] **7. Training pipeline.** Point-in-time joins, time-based splits, retraining cadence, versioned data and models.
- [ ] **8. Serving.** Batch or online. Latency budget per stage, caching, timeouts, fallbacks.
- [ ] **9. Evaluate and launch.** Offline gate → shadow → A/B with guardrails → staged ramp.
- [ ] **10. Monitor and iterate.** Feature freshness and nulls, prediction drift, calibration, per-segment metrics, retraining triggers.

## Recommendation funnel

| Stage | Items | Model | Optimizes |
|---|---|---|---|
| Candidate generation | 10M → ~1,000 | two-tower embeddings + ANN, plus heuristic sources | recall, cheap per item |
| Ranking | ~1,000 → ~100 | GBDT or multi-task DNN on rich and cross features | precision at the top |
| Re-ranking | ~100 → ~20 | rules and slate logic | diversity, freshness, policy, ads |

**Latency:** $L = L_{\text{overhead}} + \sum_s (f_s + n_s t_s)$. **Compute:** $\sum_s n_s t_s$ worker-ms per request; × QPS / 1000 = busy workers.

**Recalibrate after keeping a fraction $w$ of negatives:** $p = q / (q + (1-q)/w)$. Example: $q = 0.5$, $w = 0.1$ gives $p = 0.0909$.

## Data pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Training-serving skew | offline win, online loss | one feature definition; log served features and train on them |
| Point-in-time leakage | suspiciously high offline AUC | as-of joins; time-based splits |
| Label leakage | a feature that is a consequence of the label | audit features against the label's timeline |
| Position bias | top slots look great regardless of item | position as a feature, randomization, IPW |
| Feedback loop | diversity collapses, new items starve | exploration traffic, freshness boosts |
| Cold start | new users or items get poor results | content features, popularity fallback, exploration |
| Delayed labels | recent data looks like fewer positives | wait for label maturity or model the delay |

## A/B testing

| Quantity | Formula |
|---|---|
| z-test | $z = (\hat{p}_B - \hat{p}_A) / \sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}$, $\hat{p}$ pooled |
| Sample size per arm | $n = (z_{1-\alpha/2} + z_{1-\beta})^2 [p_A(1-p_A) + p_B(1-p_B)] / \delta^2$ |
| Shortcut (α 0.05, power 0.8) | $n \approx 16\,p(1-p)/\delta^2$ |
| SRM | $\chi^2 = \sum (\text{obs} - \text{exp})^2 / \text{exp}$, 1 degree of freedom |
| CUPED | $Y - \theta(X - \bar{X})$, variance × $(1 - \rho^2)$ |

**Reference numbers:** $z_{0.975} = 1.960$, $z_{0.8} = 0.8416$. Baseline 10%, relative MDE 10%: 14,749 per arm.

**Traps:** peeking (fix the horizon or use sequential tests) · SRM (block the readout) · novelty and primacy (run long enough, plot the daily effect) · interference (cluster or switchback randomization) · many metrics (pre-register one primary metric).

Lessons: [ML system design](../lessons/22-ml-system-design/01-ml-system-design.md) · [Experimentation and A/B testing](../lessons/22-ml-system-design/02-experimentation-and-ab-testing.md) · [Production architecture](../lessons/20-production-ai/01-production-architecture.md)
