# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/21-feature-stores-and-training-serving-skew, phases/02-ml-fundamentals/19-ml-system-design
"""Feature definitions, point-in-time joins, training-serving skew detection, and a baseline gate, from scratch.

All events and users are SYNTHETIC (seeded). Stdlib only.
Run from guide/:  uv run code/22-ml-system-design/feature_skew_scratch.py
"""
import bisect
import random
from collections import defaultdict


# --- 1. One feature definition, used by both training and serving ---------------------------------
def days_since_last(days, as_of):
    """Days since the user's last purchase, seeing only events on or before `as_of`."""
    seen = [d for d in days if d <= as_of]
    return float(as_of - max(seen)) if seen else 365.0


def purchases_30d(days, as_of):
    return float(sum(1 for d in days if as_of - 30 < d <= as_of))


# --- 2. Synthetic users: half stop buying at a random day (they churn) ----------------------------
def make_events(n_users=300, horizon=180, seed=11):
    rng = random.Random(seed)
    events = defaultdict(list)                       # user -> sorted purchase days
    for u in range(n_users):
        rate, stop = rng.uniform(0.08, 0.35), (rng.randint(40, 170) if rng.random() < 0.5 else None)
        for day in range(horizon):
            if stop is not None and day >= stop:
                break
            if rng.random() < rate:
                events[f"u{u:03d}"].append(day)
    return events


def make_labels(events, label_days, window=30):
    """Label = 1 (churned) if the user makes no purchase in the 30 days AFTER the label day."""
    return [(u, t, 0 if any(t < d <= t + window for d in days) else 1)
            for u, days in sorted(events.items()) for t in label_days if any(d <= t for d in days)]


# --- 3. Offline feature table: one snapshot per user per week --------------------------------------
def backfill(events, snapshot_days):
    return {u: [(t, days_since_last(days, t)) for t in snapshot_days] for u, days in events.items()}


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


# --- 4. A one-feature rule and its accuracy --------------------------------------------------------
def fit_threshold(rows):
    best = max(range(120), key=lambda th: sum((x > th) == y for x, y in rows))
    return best, sum((x > best) == y for x, y in rows) / len(rows)


# --- 5. Skew: compare logged serving values with offline values for the SAME requests ---------------
def legacy_purchases_30d(days, as_of):
    return float(sum(1 for d in days if as_of - 30 <= d <= as_of))   # bug: 31-day window (<= vs <)


def skew_rate(events, requests, offline_fn, online_fn, tol=1e-9):
    diffs = [abs(offline_fn(events[u], t) - online_fn(events[u], t)) for u, t in requests]
    return sum(d > tol for d in diffs) / len(diffs), sum(diffs) / len(diffs)


# --- 6. Baseline gate: a model only ships if it beats the best trivial baseline by a margin ---------
def beats_baselines(model_acc, baseline_accs, min_lift=0.02):
    best = max(baseline_accs, key=baseline_accs.get)
    return best, model_acc - baseline_accs[best], model_acc - baseline_accs[best] >= min_lift


if __name__ == "__main__":
    events = make_events()
    snapshot_days = list(range(7, 176, 7))
    train_days, test_days = list(range(35, 120, 7)), list(range(126, 150, 7))
    table = backfill(events, snapshot_days)
    train, test = make_labels(events, train_days), make_labels(events, test_days)
    print(f"synthetic: {len(events)} users, {len(train)} train labels, {len(test)} test labels")

    honest_th, honest_train = fit_threshold(point_in_time_join(train, table))
    leaky_th, leaky_train = fit_threshold(naive_latest_join(train, table))
    honest_test = sum((x > honest_th) == y for x, y in point_in_time_join(test, table)) / len(test)
    leaky_test = sum((x > leaky_th) == y for x, y in point_in_time_join(test, table)) / len(test)
    print(f"point-in-time join: threshold {honest_th:3d} days, train acc {honest_train:.3f}, test acc {honest_test:.3f}")
    print(f"naive latest join : threshold {leaky_th:3d} days, train acc {leaky_train:.3f}, test acc {leaky_test:.3f}")

    rng = random.Random(3)
    requests = [(rng.choice(sorted(events)), rng.randint(40, 175)) for _ in range(500)]
    rate, mad = skew_rate(events, requests, purchases_30d, purchases_30d)
    print(f"skew, shared definition   : mismatch rate {rate:.3f}, mean |diff| {mad:.3f}")
    rate, mad = skew_rate(events, requests, purchases_30d, legacy_purchases_30d)
    print(f"skew, legacy 31-day window: mismatch rate {rate:.3f}, mean |diff| {mad:.3f}  -> flag (> 0.01)")

    churn_rate = sum(y for _, _, y in test) / len(test)
    baselines = {"majority class": max(churn_rate, 1 - churn_rate)}
    best, lift, ok = beats_baselines(honest_test, baselines)
    print(f"baseline gate: best baseline '{best}' acc {baselines[best]:.3f}, lift {lift:+.3f} -> ship={ok}")

    assert leaky_test < honest_test - 0.1, "the leaky threshold should collapse on honest test features"
    assert skew_rate(events, requests, purchases_30d, purchases_30d)[0] == 0.0
