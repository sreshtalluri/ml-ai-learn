---
title: ML system design
summary: Use a repeatable ten-step framework to design an ML system in an interview or at work, and apply it end to end to a multi-stage recommendation feed with a latency budget.
skill: engineering
minutes: 50
prerequisites: [ml-workflow, classification-metrics, random-forests-and-boosting]
adapted_from: [phases/02-ml-fundamentals/19-ml-system-design, phases/02-ml-fundamentals/21-feature-stores-and-training-serving-skew]
related: [experimentation-and-ab-testing, word-embeddings, production-architecture, reliability-cost-and-observability]
---

# ML system design

> **Mental model.** An ML system is a factory, not a model. Raw events come in, get turned into labels and features, feed a model, and the model's outputs go back out to users, whose reactions become the next day's raw events. Designing the system means designing every station on that loop and the budget each one gets, not just picking the fanciest model.

**You will learn to**
- Run a ten-step design framework, from clarifying the goal to monitoring, and spend 45 interview minutes on it sensibly.
- Separate business, offline, and online metrics, and explain why they disagree.
- Design a recommendation funnel (two-tower candidate generation with ANN search, a feature-rich ranker, a re-ranker) and size it against a latency budget.
- Spot and prevent training-serving skew, point-in-time leakage, position bias, and feedback loops.
- Correct predicted probabilities after negative downsampling.

**Why it matters.** "Design a feed / fraud / search system" is the standard ML engineer interview, and the same reasoning decides whether a real project ships. Most ML failures in production are system failures: a feature computed differently online than offline, a label that leaks the future, a metric that rewards the wrong thing. A strong design names those risks before they happen.

## 1. Intuition

### The framework

Every ML design question, whatever the product, can be answered with the same ten steps. Interviewers grade the *order* as much as the content: jumping to "I'd use a transformer" before you know the goal is the most common way to fail.

| # | Step | The question you answer |
|---|---|---|
| 1 | **Clarify goal and constraints** | What does the business want? Scale (users, items, QPS)? Latency? Freshness? Fairness, privacy, or regulatory limits? |
| 2 | **Frame the ML task** | What exactly is predicted, for whom, and when? Classification, regression, ranking, retrieval? |
| 3 | **Metrics** | Business metric (revenue, retention), offline metric (AUC, nDCG, recall@k), online metric (CTR, watch time, complaints), and guardrails. |
| 4 | **Data and labels** | Where do labels come from (explicit, implicit, human)? How delayed and noisy are they? Is there bias in how they were collected? |
| 5 | **Features** | User, item, context, and interaction features. Which are real-time, which are batch? Can they be computed identically offline and online? |
| 6 | **Model, baseline first** | A heuristic or popularity baseline, then a simple model (logistic regression, GBDT), then deep models only if they earn their cost. |
| 7 | **Training pipeline** | Data snapshots, point-in-time joins, train/validation split by time, retraining cadence, reproducibility. |
| 8 | **Serving** | Batch or online? Latency budget per stage, caching, fallbacks, hardware. |
| 9 | **Evaluation and launch** | Offline evaluation, shadow mode, A/B test with guardrails, ramp-up. |
| 10 | **Monitoring and iteration** | Data and prediction [drift](../../glossary.md#drift), feature freshness, metric dashboards, alerting, retraining triggers, the next experiment. |

### Spending 45 minutes

| Minutes | Steps | What a good answer sounds like |
|---|---|---|
| 0 to 5 | 1 | Asks 3 to 5 clarifying questions, states assumptions out loud ("say 100M daily users, 10k QPS peak, 200 ms p99") |
| 5 to 10 | 2, 3 | Writes down the prediction target and the three metric layers before any model talk |
| 10 to 17 | 4, 5 | Names the label source and its biases; sketches feature groups and which need a feature store |
| 17 to 25 | 6 | Baseline, then the real design; justifies each stage with a number |
| 25 to 32 | 7, 8 | Training data flow; serving path with a latency budget; batch versus online |
| 32 to 40 | 9, 10 | Offline gate, A/B plan, monitoring, what could go wrong |
| 40 to 45 | | Deep dive wherever the interviewer pushes; summarize trade-offs |

> [!TIP]
> Draw the diagram early (boxes for data sources, feature pipeline, training, model registry, serving, logging) and keep pointing at it. It turns a ramble into a tour.

### The worked case: a recommendation feed

**Clarify.** A short-video app wants a personalized home feed. Assume 10M items in the catalog, 10,000 requests per second at peak, a 150 ms server-side budget at p99, and 20 items returned per request. Goal: long-term engagement (daily active users), not just clicks.

**Frame.** For a (user, context) pair, rank the catalog and return the top 20. In practice that is a ranking problem solved as pointwise prediction: estimate $P(\text{engage} \mid \text{user}, \text{item}, \text{context})$ for several engagement types (click, watch past 50%, like, share, hide) and combine them into one score with business weights.

**Metrics.**

| Layer | Examples | Why it exists |
|---|---|---|
| Business | 7-day retention, daily active users, session time | what the company actually wants; slow and noisy |
| Online (A/B) | watch time per user, CTR, hides and reports per 1,000 impressions | measured on real traffic in days |
| Offline | AUC and log loss per head, recall@k for candidate generation, nDCG@20 | measured in minutes on logged data; decides what earns an A/B slot |
| Guardrails | p99 latency, error rate, creator diversity, policy-violating content rate | must not get worse even if the main metric improves |

**Data and labels.** Labels are implicit: impressions joined with later clicks, watches, likes, and hides. They arrive with a delay (a watch-time label is final only when the session ends), they only exist for items the old system chose to show (selection bias), and they are skewed by where the item was shown ([position bias](../../glossary.md#position-bias)).

**Features.** User (history embeddings, long-term topic affinities, account age), item (content embedding, creator, age, recent engagement rates), context (time of day, device, network), and cross features (user's past engagement with this creator). Real-time counters such as "views of this item in the last hour" come from a streaming job and a [feature store](../../glossary.md#feature-store).

**Model: a three-stage funnel.** You cannot run a rich model over 10M items in 150 ms, so the system narrows the catalog in stages, each one more expensive per item and run on fewer items.

1. **Candidate generation (retrieval).** A [two-tower model](../../glossary.md#two-tower-model) maps the user and each item to vectors; an approximate nearest neighbor (ANN) index returns the ~1,000 items whose vectors have the highest dot product with the user vector. Several sources usually run in parallel and are merged: two-tower, "more from creators you follow", trending in your region, fresh uploads. Goal: high recall, cheap per item.
2. **Ranking.** A gradient-boosted tree model or a deep network (often multi-task, one head per engagement type) scores each of the ~1,000 candidates using hundreds of features, including cross features the two-tower model cannot express. Goal: precision at the top.
3. **Re-ranking.** Business logic and slate-level objectives on the top ~100: diversity (no five videos from one creator in a row), freshness boosts, deduplication, policy filters, ads insertion. Returns the final 20.

Baseline first: before any of this, "most popular in your region in the last 24 hours" is a real system that ships in a day and gives every later model something to beat.

**Training pipeline.** Daily (or hourly) jobs build training examples by joining logged impressions with labels and with the feature values *as they were at impression time*. Split train and validation by time, never randomly. Retrain the ranker daily; refresh the item tower and ANN index more often than the user tower when new items arrive constantly.

**Serving.** Online, because the feed depends on what the user did a minute ago. Precompute item embeddings in batch, compute the user embedding per request, cache aggressively, and set a latency budget per stage (worked out in section 3). If ranking times out, fall back to the candidate generator's order; if everything fails, serve the popularity baseline.

**Evaluation and launch.** Offline gate (AUC, log loss, and nDCG on the most recent days), then a shadow deployment to check latency and score distributions, then an A/B test on watch time with guardrails (see [Experimentation and A/B testing](02-experimentation-and-ab-testing.md)), then a staged ramp.

**Monitoring.** Feature freshness and null rates, prediction distribution versus training, [calibration](../../glossary.md#calibration) per head, per-stage latency, and the online metrics by segment (new users, new items, regions).

### Two more cases in one table

| Decision | Feed recommendation | Fraud detection | Search ranking |
|---|---|---|---|
| Prediction | $P(\text{engage})$ per item, several heads | $P(\text{fraud})$ per transaction | relevance of a document to a query |
| Labels | implicit, plentiful, biased by exposure | chargebacks and analyst reviews; weeks of delay; ~0.1% positive | human relevance ratings plus clicks (position-biased) |
| Key metric | watch time, retention | recall at a fixed false-positive rate, dollars of fraud caught | nDCG@10, query success rate, abandonment |
| Serving | online funnel, ~100 to 200 ms | synchronous scoring in the payment path, often under 50 ms, with a rules engine alongside | retrieval (inverted index plus vectors) then a learned-to-rank model |
| Model | two-tower plus GBDT or DNN ranker | GBDT on aggregates (velocity counts, device and graph features) | BM25 plus dense retrieval, then a cross-encoder or GBDT ranker |
| Special risk | feedback loops, popularity bias | adversaries adapt, so drift is intentional; label delay | queries with no clicks, freshness for news |

## 2. Visualization

<!-- lab:recsys-funnel -->
![Left: a horizontal bar showing 20 ms overhead, 17 ms candidate generation, 75 ms ranking, and 13 ms re-ranking, totalling 125 ms against a dashed 150 ms budget line. Right: as the number of candidates passed to the ranker rises from 100 to 5,000, total latency grows in a straight line and crosses the 150 ms budget near 1,400 candidates, while synthetic end-to-end recall flattens out at about 0.75 after roughly 1,000 candidates.](../../figures/ml-system-design.png)

*Synthetic cost model. Left: the worked latency budget from section 3. Right: passing more candidates to the ranker buys almost no recall after about 1,000 but keeps costing latency linearly, which is why the funnel narrows aggressively.*

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/recsys-funnel/).*
<!-- /lab -->

**Try it** (predict first, then check):

1. Click **Rank 5,000**. Predict whether recall goes up a lot or a little, and by how many milliseconds you overshoot the budget.
2. Click **Starved ranker** (only 200 candidates). Latency drops; which stage now loses the most relevant items?
3. Click **Heavy ranker** (0.2 ms per item). Find the largest k1 that still fits 150 ms, and compare its recall with the default.
4. Set the budget to 100 ms. Is it cheaper in recall to cut k1 or to cut k2?

## 3. The math

### Symbols

| Symbol | Meaning | Shape / unit |
|---|---|---|
| $N$ | catalog size | items |
| $k_1, k_2, k_3$ | items kept by candidate generation, ranking, re-ranking | items |
| $t_s$ | per-item cost of stage $s$ | ms per item |
| $f_s$ | fixed cost of stage $s$ (feature fetch, RPC) | ms |
| $L$, $B$ | total latency and latency budget | ms |
| $\mathbf{u}$ | user tower output | $[d]$ |
| $\mathbf{V}$ | item tower outputs for the catalog | $[N, d]$ |
| $\tau_s$ | synthetic recall scale of stage $s$ | items |
| $w$ | fraction of negatives kept when downsampling | scalar in $(0, 1]$ |
| $q$, $p$ | model output after downsampling, calibrated probability | scalars |

### Two-tower scoring

```math
\mathbf{u} = f_\theta(\text{user, context}) \in \mathbb{R}^{d}, \qquad \mathbf{v}_i = g_\phi(\text{item}_i) \in \mathbb{R}^{d}, \qquad s(u, i) = \mathbf{u} \cdot \mathbf{v}_i
```

Because the score is a dot product, all $N$ item vectors $\mathbf{V}$ can be computed offline and put in an ANN index (graph-based such as HNSW, or quantized such as IVF-PQ). At request time you compute one user vector and ask the index for the top $k_1$ by dot product, in roughly logarithmic rather than linear time in $N$. The price: user and item interact only through one dot product, so the model cannot express cross features like "this user likes this creator, but only at night". That is the ranker's job.

Training typically uses in-batch negatives with a softmax over the batch. Popular items appear as negatives more often than they should, so a common fix subtracts $\log Q(i)$ (the item's sampling probability) from its logit during training.

### Funnel cost model

```math
L = L_{\text{overhead}} + \sum_{s=1}^{3} \left( f_s + n_s \, t_s \right), \qquad n_1 = k_1,\; n_2 = k_1,\; n_3 = k_2
```

where $n_s$ is the number of items stage $s$ processes. Candidate generation pays per returned candidate, ranking scores all $k_1$ candidates, and re-ranking scores the top $k_2$. Compute per request (busy worker-milliseconds) is $\sum_s n_s t_s$. This is a one-worker model with no sharding; real systems parallelize, which lowers latency but not compute.

For the lab, each stage keeps relevant items with a synthetic saturating curve, and end-to-end recall is the product:

```math
r_s = 1 - e^{-k_s / \tau_s}, \qquad R = r_1 \, r_2 \, r_3
```

### Worked example 1: the latency budget

Constants (synthetic): overhead 20 ms; candidate generation $f_1 = 12$, $t_1 = 0.005$; ranking $f_2 = 15$, $t_2 = 0.06$; re-ranking $f_3 = 3$, $t_3 = 0.1$. Keep $k_1 = 1{,}000$, $k_2 = 100$, $k_3 = 20$. Budget $B = 150$ ms.

| Stage | Items processed | Latency | Stage recall |
|---|---|---|---|
| Candidate generation | 1,000 returned | $12 + 1000 \times 0.005 = 12 + 5 = 17$ ms | $1 - e^{-1000/250} = 1 - e^{-4} = 1 - 0.0183 = 0.9817$ |
| Ranking | 1,000 scored | $15 + 1000 \times 0.06 = 15 + 60 = 75$ ms | $1 - e^{-100/50} = 1 - e^{-2} = 1 - 0.1353 = 0.8647$ |
| Re-ranking | 100 scored | $3 + 100 \times 0.1 = 3 + 10 = 13$ ms | $1 - e^{-20/10} = 1 - e^{-2} = 0.8647$ |

Total: $L = 20 + 17 + 75 + 13 = 125$ ms. Headroom: $150 - 125 = 25$ ms. End-to-end recall: $R = 0.9817 \times 0.8647 \times 0.8647 = 0.8489 \times 0.8647 = 0.7340$.

Now double the candidates to $k_1 = 2{,}000$: candidate generation becomes $12 + 10 = 22$ ms and ranking $15 + 120 = 135$ ms, so $L = 20 + 22 + 135 + 13 = 190$ ms, 40 ms over budget, while $r_1$ only moves from 0.9817 to $1 - e^{-8} = 0.9997$.

### Worked example 2: why not rank everything?

Ranking the full catalog: $10{,}000{,}000 \times 0.06 = 600{,}000$ ms $= 600$ s per request. The funnel exists because that number is four orders of magnitude over budget.

### Worked example 3: capacity

Compute per request: $1000 \times 0.005 + 1000 \times 0.06 + 100 \times 0.1 = 5 + 60 + 10 = 75$ worker-ms. At 10,000 QPS: $75 \times 10{,}000 = 750{,}000$ worker-ms per second $= 750$ worker-seconds per second, so about 750 workers fully busy, before headroom for peaks and failures. Halving the ranker's per-item cost (distillation, fewer features) saves 30 worker-ms per request, which is 300 workers.

### Worked example 4: calibration after negative downsampling

Clicks are rare, so training sets often keep all positives and only a fraction $w$ of negatives. That multiplies the odds of a positive by $1/w$, so the model's output $q$ is too high. Undo it in odds space:

```math
\frac{p}{1-p} = w \cdot \frac{q}{1-q} \quad\Longleftrightarrow\quad p = \frac{q}{q + (1-q)/w}
```

With $w = 0.1$ (keep 10% of negatives) and $q = 0.5$: odds $= 0.5/0.5 = 1$, times $w$ gives $0.1$, so $p = 0.1 / 1.1 = 0.0909$. Check with the closed form: $0.5 / (0.5 + 0.5/0.1) = 0.5 / (0.5 + 5) = 0.5 / 5.5 = 0.0909$. With $q = 0.2$: $0.2 / (0.2 + 0.8/0.1) = 0.2 / 8.2 = 0.0244$.

Ranking order is unchanged by this correction (it is monotonic), so it matters only when the probability itself is used: ad auctions (bid $\times$ pCTR), combining several heads with weights, or thresholds.

## 4. Implementation

Two of the system failures named above, point-in-time leakage and training-serving skew, are invisible in a design diagram and obvious in code. This section builds them from scratch on synthetic purchase data, then shows the production version.

### Build it from scratch

The script is [`code/22-ml-system-design/feature_skew_scratch.py`](../../code/22-ml-system-design/feature_skew_scratch.py) (stdlib only, seeded synthetic users). The task: predict whether a user churns (no purchase in the next 30 days).

**One feature definition.** The function takes `as_of` and only looks at events on or before it. Every feature in a real feature store has this shape: a definition plus a timestamp. Training and serving both call the same function, which is the whole defense against skew.

```python
def days_since_last(days, as_of):
    """Days since the user's last purchase, seeing only events on or before `as_of`."""
    seen = [d for d in days if d <= as_of]
    return float(as_of - max(seen)) if seen else 365.0

def purchases_30d(days, as_of):
    return float(sum(1 for d in days if as_of - 30 < d <= as_of))
```

*Predict:* a user with no purchases yet gets 365. What would happen to the model if offline used 365 but the online service used 0 for the same case?

**Labels look into the future, on purpose.** A label at day `t` asks about days `t+1` to `t+30`. That is fine for a label. It becomes a leak only if a feature also sees past `t`.

```python
def make_labels(events, label_days, window=30):
    """Label = 1 (churned) if the user makes no purchase in the 30 days AFTER the label day."""
    return [(u, t, 0 if any(t < d <= t + window for d in days) else 1)
            for u, days in sorted(events.items()) for t in label_days if any(d <= t for d in days)]
```

*Predict:* if you computed `days_since_last` at day `t + 30` instead of `t`, what value would almost every churned user get, and why would accuracy look perfect?

**Two ways to join features to labels.** The offline feature table holds one snapshot per user per week. The point-in-time join takes, for each label, the latest snapshot at or before the label day (binary search over the snapshot days). The naive join takes the user's latest snapshot, which for old labels is from the future.

```python
def point_in_time_join(labels, table):
    """For each label at day t, take the latest snapshot with day <= t. Never a later one."""
    out = []
    for u, t, y in labels:
        snaps = table[u]
        i = bisect.bisect_right([d for d, _ in snaps], t) - 1
        out.append((snaps[i][1] if i >= 0 else 365.0, y))
    return out

def naive_latest_join(labels, table):
    """The classic leak: join every label to the user's LATEST snapshot, which can be from the future."""
    return [(table[u][-1][1], y) for u, t, y in labels]
```

*Predict:* the naive join's training accuracy is about the same as the honest one here. Why can a leak hurt badly even when it doesn't inflate the training score? (Look at the threshold each one learns in the output below.)

**Skew detection.** Replay the same (user, time) requests through the offline definition and the online one, and count disagreements. Here the "online" path is a legacy reimplementation with `<=` instead of `<`, a 31-day window. Nothing crashes; the numbers are just quietly different.

```python
def legacy_purchases_30d(days, as_of):
    return float(sum(1 for d in days if as_of - 30 <= d <= as_of))   # bug: 31-day window (<= vs <)

def skew_rate(events, requests, offline_fn, online_fn, tol=1e-9):
    diffs = [abs(offline_fn(events[u], t) - online_fn(events[u], t)) for u, t in requests]
    return sum(d > tol for d in diffs) / len(diffs), sum(diffs) / len(diffs)
```

*Predict:* the mismatch rate is the fraction of requests with a purchase exactly 30 days before. Roughly what rate would you expect if users buy on about 20% of days?

**A baseline gate.** Step 6 of the framework as code: a model ships only if it beats the best trivial baseline by a margin you set in advance.

```python
def beats_baselines(model_acc, baseline_accs, min_lift=0.02):
    best = max(baseline_accs, key=baseline_accs.get)
    return best, model_acc - baseline_accs[best], model_acc - baseline_accs[best] >= min_lift
```

*Predict:* on a dataset that is 95% "no churn", what accuracy does the majority-class baseline get, and why does that make accuracy a poor primary metric there?

### Use the library

In production the definitions live in a [feature store](../../glossary.md#feature-store) such as Feast, and the same registry serves both paths. Feast's offline `get_historical_features` does the point-in-time join against an entity table of (entity, timestamp) rows; `get_online_features` reads the latest materialized values for serving:

```python
from feast import FeatureStore  # pip install feast

store = FeatureStore(repo_path=".")
train_df = store.get_historical_features(          # point-in-time join, offline
    entity_df=labels_df,                            # columns: user_id, event_timestamp, label
    features=["user_stats:days_since_last", "user_stats:purchases_30d"],
).to_df()
online = store.get_online_features(                 # low-latency read, serving
    features=["user_stats:days_since_last", "user_stats:purchases_30d"],
    entity_rows=[{"user_id": "u042"}],
).to_dict()
```

The funnel that consumes these features is a chain of narrowing calls, each with its own timeout and fallback:

```python
def recommend(user_id, ctx, k1=1000, k2=100, k3=20):
    u = user_tower(feature_store.get_user(user_id), ctx)                  # [d]
    cands = merge(ann_index.search(u, k1), follows(user_id), trending(ctx.region))
    feats = feature_store.get_items([c.id for c in cands], user_id, ctx)  # [k1, n_features]
    try:
        scores = ranker.predict(feats, timeout_ms=80)                     # [k1, n_heads]
    except Timeout:
        scores = [c.retrieval_score for c in cands]                        # fallback: retrieval order
    top = top_k(cands, combine(scores, weights=HEAD_WEIGHTS), k2)
    return rerank(top, rules=[dedupe, diversify_by_creator, policy_filter], k=k3)
```

The single most important line is `feature_store.get_items`: the same feature definitions must produce the training data, or the model will see different numbers in production than it learned on. The library does not remove the need for the skew check above. Log the served feature vector with each prediction and compare it with the offline value for the same request.

### Run it

From `guide/`:

```bash
uv run code/22-ml-system-design/feature_skew_scratch.py
```

```text
synthetic: 300 users, 3895 train labels, 1200 test labels
point-in-time join: threshold  18 days, train acc 0.912, test acc 0.911
naive latest join : threshold  97 days, train acc 0.914, test acc 0.646
skew, shared definition   : mismatch rate 0.000, mean |diff| 0.000
skew, legacy 31-day window: mismatch rate 0.188, mean |diff| 0.188  -> flag (> 0.01)
baseline gate: best baseline 'majority class' acc 0.617, lift +0.294 -> ship=True
```

The leaky join trains to the same accuracy but learns a 97-day threshold, because its features come from the end of the data where churned users have been idle for months. Scored on honest features, that rule falls to 0.646, barely above the 0.617 baseline. Offline metrics did not warn you; only a time-correct evaluation did.

The latency, capacity, and recall arithmetic from section 3, the calibration correction, and this lesson's figure are in [`code/22-ml-system-design/recsys_funnel.py`](../../code/22-ml-system-design/recsys_funnel.py).

## 5. Engineering

**Training-serving skew.** [Training-serving skew](../../glossary.md#training-serving-skew) is any difference between the features or preprocessing seen at training time and at serving time: a feature computed in SQL offline and in Java online, a different tokenizer version, a missing-value default of 0 offline and -1 online, a counter that is hourly offline and real-time online. Fixes: define each feature once in a [feature store](../../glossary.md#feature-store) that serves both paths; log the exact feature vector served with each prediction and train on those logs; compare online and offline feature distributions continuously.

**Point-in-time correctness.** A training example for an impression at 10:00 must use feature values as of 10:00. Joining today's "user's total likes" onto last month's impressions leaks the future: the label (did they like it?) is partly inside the feature. Feature stores provide point-in-time ("as-of") joins for exactly this reason.

**Label leakage.** Beyond time travel, watch for features that are consequences of the label: "session length" when predicting a watch, "refund issued" when predicting fraud, an ID that encodes the outcome. A sudden offline AUC of 0.99 is a [data leakage](../../glossary.md#data-leakage) alarm, not a celebration.

**Position bias and feedback loops.** Users click the top slot partly because it is the top slot. A model trained naively on clicks learns "whatever we showed first is good", and then shows it first again: a feedback loop that entrenches popular items and starves new ones. Mitigations: add position as a training feature and set it to a fixed value at serving time, learn propensities by occasionally randomizing positions (inverse propensity weighting), and reserve a small exploration slice of traffic.

**Cold start.** [Cold start](../../glossary.md#cold-start) is the lack of interaction history for new users or items. New items: rely on content features (embeddings of the video, title, creator) in the item tower, plus a freshness boost or exploration budget in re-ranking. New users: use context (country, device, sign-up answers) and popular items, then adapt quickly from the first few interactions with real-time features.

**Class imbalance and negative sampling.** With CTRs near 1%, keep all positives and downsample negatives, then recalibrate (worked example 4). For the two-tower model, "negatives" are usually other items in the batch or random catalog items; hard negatives (items shown but skipped) sharpen the model but must be mixed with easy ones.

**Calibration.** Multi-task rankers combine heads as $\sum_h \beta_h \hat{p}_h$; that sum is only meaningful if each $\hat{p}_h$ is a real probability. Monitor calibration (predicted versus observed rate per score bucket) per head and per segment, and recalibrate with isotonic or Platt scaling on recent data.

**Offline-online mismatch.** Offline metrics are computed on logs collected by the old system, so they can only judge items the old system chose to show (selection bias); they ignore position effects, latency, and slate interactions; and they measure short-term clicks, not long-term retention. Expect some offline wins to lose online. Track how well offline deltas have predicted online deltas over past launches, and use that history to decide how much to trust a new offline result.

**Batch versus online prediction.** If recommendations can be a few hours stale (a weekly email, "items you might like" on a low-traffic page), precompute them in batch: cheaper and simpler. Use online inference when the context changes per request (search queries, the current session, fraud at payment time).

> [!WARNING]
> **Failure modes.** A feature pipeline silently returns defaults after an upstream schema change; the ANN index is rebuilt with a new item tower but the old user tower (vectors from different models are incompatible); a popularity feedback loop collapses diversity; delayed labels make yesterday's data look like it has fewer positives, so a daily retrain learns a lower base rate.

### Common mistakes

- Naming a model before stating the goal, the prediction target, and the metric.
- Splitting train and validation randomly on time-ordered data, which leaks the future.
- Optimizing the offline metric only and calling it done, with no plan for online evaluation or guardrails.
- Forgetting that downsampled training data produces miscalibrated probabilities.
- Ignoring the latency budget: proposing a cross-encoder over 10M items.
- No baseline, so nobody knows whether the complex system is worth its cost.

## 6. Knowledge check

<!-- quiz:ml-system-design -->
**[Take the ML system design quiz](../../quizzes/ml-system-design.md)**
<!-- /quiz -->

**Practice exercise.** Your feed ranker scores 1,500 candidates at 0.04 ms each with 15 ms fixed cost. Candidate generation costs $12 + 1500 \times 0.005$ ms, re-ranking scores 100 items at 0.1 ms each plus 3 ms, and overhead is 20 ms. Does the system fit a 150 ms budget, and how many workers does it need at 10,000 QPS?

<details>
<summary>Solution</summary>

Candidate generation: $12 + 1500 \times 0.005 = 12 + 7.5 = 19.5$ ms. Ranking: $15 + 1500 \times 0.04 = 15 + 60 = 75$ ms. Re-ranking: $3 + 100 \times 0.1 = 13$ ms. Total: $20 + 19.5 + 75 + 13 = 127.5$ ms, so it fits with 22.5 ms of headroom. Compute: $7.5 + 60 + 10 = 77.5$ worker-ms per request; at 10,000 QPS that is $77.5 \times 10{,}000 / 1000 = 775$ busy workers. A cheaper ranker let you pass 50% more candidates for about the same cost.
</details>

**Implementation challenge.** Extend `recsys_funnel.py` with a fourth stage, a "light ranker" between candidate generation and the heavy ranker (for example, 5,000 candidates in, 500 out, 0.004 ms per item). Find a configuration that raises synthetic recall above the three-stage default while staying under 150 ms.

<details>
<summary>Solution sketch</summary>

Add a stage with its own $(f, t, \tau)$, let candidate generation return 5,000 items (cost $12 + 5000 \times 0.005 = 37$ ms), let the light ranker score them ($f + 5000 \times 0.004 = f + 20$ ms) and keep 500, and let the heavy ranker score 500 instead of 1,000 ($15 + 500 \times 0.06 = 45$ ms). With $f = 5$: $20 + 37 + 25 + 45 + 13 = 140$ ms. Candidate recall rises to $1 - e^{-5000/250} \approx 1.0$; whether the total rises depends on the light ranker's $\tau$. The pattern (cascade of ever more expensive models on ever fewer items) is exactly why production systems often have four or more stages.
</details>

## Summary

- Answer every design question in the same order: goal and constraints, task framing, metrics, data, features, baseline then model, training, serving, launch, monitoring.
- Keep three metric layers apart (business, online, offline) plus guardrails, and expect offline and online results to disagree sometimes.
- Recommendation systems are funnels: cheap high-recall retrieval (two-tower plus ANN), an expensive precise ranker, then re-ranking for diversity and business rules, each sized against a latency budget.
- Most production failures are data failures: training-serving skew, point-in-time leakage, position bias and feedback loops, delayed labels.
- Downsampling negatives requires recalibration: $p = q / (q + (1-q)/w)$.

**Next:** [Experimentation and A/B testing](02-experimentation-and-ab-testing.md)

**Related:** [ML workflow](../02-ml-workflow/01-ml-workflow.md) · [Classification metrics](../04-classification/02-classification-metrics.md) · [Random forests and boosting](../06-trees-and-ensembles/02-random-forests-and-boosting.md) · [Production architecture](../20-production-ai/01-production-architecture.md) · [Cheat sheet: ML system design](../../cheatsheets/ml-system-design.md) · [Model card: two-tower model](../../models/two-tower-model.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Graph Theory for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/21-graph-theory) · [Design Success Metrics Before the Result Exists](https://aiengineeringfromscratch.com/lesson?path=phases/14-agent-engineering/52-design-success-metrics) · [Model, System, and Dataset Cards](https://aiengineeringfromscratch.com/lesson?path=phases/18-ethics-safety-alignment/26-model-system-dataset-cards)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [ML System Design — From Problem to Deployed Model](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/19-ml-system-design), [Feature Stores and Training-Serving Skew](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/21-feature-stores-and-training-serving-skew).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Why do recommendation systems use a multi-stage funnel instead of one big model?</strong></summary>

Because no single model can be both accurate and cheap enough to score millions of items within a ~100 to 200 ms budget. The funnel spends compute where it matters. Candidate generation, usually a two-tower model with an ANN index, cuts 10M items to about 1,000 in a few milliseconds, because item vectors are precomputed and the search is sublinear. Its job is recall. A ranker with hundreds of features and cross features then scores only those 1,000; its job is precision at the top. Re-ranking applies slate-level rules such as diversity, freshness, and policy. In numbers: ranking 10M items at 0.06 ms each would take 600 seconds per request, while ranking 1,000 takes 60 ms. The trade-off is that anything retrieval misses can never be recovered later, so you monitor candidate recall separately.

</details>

<details>
<summary><strong>When would you choose a GBDT ranker over a deep neural ranker?</strong></summary>

Choose GBDT when features are mostly tabular and hand-engineered (counts, rates, recency, categorical IDs with modest cardinality), data is in the millions rather than billions of rows, and you need fast iteration, CPU serving, and easy feature importance. Gradient-boosted trees are strong on heterogeneous tabular data with little tuning, and are a common first ranker. Choose a deep ranker when you have huge logs, high-cardinality sparse IDs that benefit from learned embeddings, raw content inputs (text, images, sequences of past actions), or several related objectives you want in one multi-task model with shared layers. The deep model costs more to train and serve, usually on GPUs, and is harder to debug. A common path is GBDT first, then a DNN once the data and team can support it, with the GBDT as the baseline it must beat online.

</details>

<details>
<summary><strong>Your new ranker beats the old one on offline AUC but loses the online A/B test. What do you check?</strong></summary>

First, training-serving skew. Log the features served online for a sample of requests and compare them with the offline features for the same impressions; a default value or a stale counter often explains everything. Second, leakage. Check that offline features were point-in-time correct, since a leaked feature inflates offline AUC and vanishes online. Third, latency. A slower model can time out and fall back to retrieval order, or slow the page enough to hurt engagement, so check p99 and fallback rates per arm. Fourth, the metric gap. AUC measures pairwise ordering on items the old system chose to show, not the top-20 slate, and it ignores position bias and long-term effects. Compare nDCG@20 and calibration, and segment the A/B results. Finally, run sanity checks on the experiment itself, such as sample ratio mismatch and guardrails.

</details>

<details>
<summary><strong>You trained a CTR model keeping 5% of negatives. It outputs 0.3 for an ad. What CTR do you use in the auction?</strong></summary>

About 2.1%. Keeping a fraction $w$ of negatives multiplies the odds of a click by $1/w$, so you multiply the predicted odds by $w$ to undo it. Predicted odds are $0.3 / 0.7 = 0.4286$. Times $w = 0.05$ gives $0.02143$. Converting back, $p = 0.02143 / 1.02143 = 0.0210$. Equivalently, $p = q / (q + (1 - q)/w) = 0.3 / (0.3 + 14) = 0.3 / 14.3 = 0.021$. Ranking by $q$ or $p$ gives the same order, so a pure ranking metric would never reveal the problem. But an auction computes bid times pCTR and compares across campaigns and models, so an uncorrected 0.3 would overcharge advertisers by more than ten times. After correcting, verify calibration on fresh unsampled traffic, predicted versus observed CTR per bucket, because the correction assumes the downsampling was uniformly random.

</details>
