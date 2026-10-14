# Adapted from AI Engineering from Scratch (MIT), phases/02-ml-fundamentals/15-time-series
"""Time series from scratch: differencing, autocorrelation, lag features, walk-forward validation, and baselines.

The series is SYNTHETIC daily demand: linear trend + weekly seasonality + noise, seeded.
Run from guide/:  uv run code/24-applied-ml/time_series.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, TEAL, plt, save  # noqa: E402


def make_series(n=730, seed=0):
    """SYNTHETIC daily demand: 100 + 0.05 * t + weekly pattern + N(0, 4) noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    weekly = np.array([-8, -2, 0, 2, 4, 10, -6])[t % 7]   # Mon..Sun effect, sums to 0
    return 100 + 0.05 * t + weekly + rng.normal(0, 4, n)


# ---------- diagnostics ----------
def difference(y, lag=1):
    """y'_t = y_t - y_{t-lag}. Loses the first `lag` points."""
    y = np.asarray(y, dtype=float)
    return y[lag:] - y[:-lag]


def acf(y, max_lag):
    """Sample autocorrelation r_k = sum (y_t - m)(y_{t-k} - m) / sum (y_t - m)^2."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    denom = np.sum(y**2)
    return np.array([np.sum(y[k:] * y[: len(y) - k]) / denom for k in range(max_lag + 1)])


# ---------- features: only the past may enter row t ----------
def lag_features(y, lags=(1, 7, 14), window=7):
    """Row t uses y[t-l] for each lag and the mean of y[t-window : t] (strictly before t)."""
    y = np.asarray(y, dtype=float)
    start = max(max(lags), window)
    rows = []
    for t in range(start, len(y)):
        rows.append([y[t - l] for l in lags] + [y[t - window:t].mean()])
    return np.array(rows), y[start:], start


def leaky_features(y, lags=(1, 7, 14), window=7):
    """BUG on purpose: same as lag_features, but the rolling mean is CENTERED (y[t-1], y[t], y[t+1])."""
    y = np.asarray(y, dtype=float)
    start = max(max(lags), window)
    rows = []
    for t in range(start, len(y) - 1):
        rows.append([y[t - l] for l in lags] + [y[t - 1:t + 2].mean()])
    return np.array(rows), y[start:len(y) - 1]


# ---------- validation and metrics ----------
def walk_forward(n, n_splits=5, test_size=60):
    """Expanding window: fold k trains on everything before its test block."""
    for k in range(n_splits):
        test_start = n - (n_splits - k) * test_size
        yield np.arange(test_start), np.arange(test_start, test_start + test_size)


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mape(y, yhat):
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def mase(y, yhat, y_train, m=1):
    """MAE scaled by the in-sample MAE of the (seasonal) naive forecast with period m."""
    scale = np.mean(np.abs(difference(y_train, m)))
    return mae(y, yhat) / scale


if __name__ == "__main__":
    print("=== Worked example (tiny series) ===")
    tiny = np.array([10, 12, 14, 13, 15, 17, 16], dtype=float)
    print(f"y            = {tiny.tolist()}")
    print(f"first diff   = {difference(tiny).tolist()}")
    print(f"acf lag 1    = {acf(tiny, 1)[1]:.4f}")
    naive = tiny[:-1]                     # forecast y_t with y_{t-1}
    print(f"naive MAE on t=1..6 = {mae(tiny[1:], naive):.4f}")
    print(f"MAPE of forecasting 16 with 17 = {mape([16], [17]):.2f}%")

    y = make_series()
    print(f"\n=== SYNTHETIC daily series: {len(y)} days ===")
    r = acf(y, 21)
    r_d = acf(difference(y, 7), 21)
    print("ACF raw      lags 1, 7, 14: " + ", ".join(f"{r[k]:.3f}" for k in (1, 7, 14)))
    print("ACF diff(7)  lags 1, 7, 14: " + ", ".join(f"{r_d[k]:.3f}" for k in (1, 7, 14)))
    half = len(y) // 2
    print(f"mean first half = {y[:half].mean():.2f}, second half = {y[half:].mean():.2f}  (trend: not stationary)")
    d7 = difference(y, 7)
    print(f"after diff(7): mean first half = {d7[:len(d7)//2].mean():.2f}, second half = {d7[len(d7)//2:].mean():.2f}")

    print("\n=== Walk-forward validation: 5 folds x 60 days, expanding window ===")
    X, target, start = lag_features(y)
    totals = {"naive (y[t-1])": [], "seasonal naive (y[t-7])": [], "ridge on lags": []}
    mase_ridge = []
    for train_idx, test_idx in walk_forward(len(target)):
        model = Ridge(alpha=1.0).fit(X[train_idx], target[train_idx])
        preds = {"naive (y[t-1])": X[test_idx, 0], "seasonal naive (y[t-7])": X[test_idx, 1],
                 "ridge on lags": model.predict(X[test_idx])}
        for k, p in preds.items():
            totals[k].append(mae(target[test_idx], p))
        mase_ridge.append(mase(target[test_idx], preds["ridge on lags"], target[train_idx], m=7))
    for k, v in totals.items():
        print(f"  {k:24s} MAE = {np.mean(v):.2f}   per fold: {', '.join(f'{x:.2f}' for x in v)}")
    print(f"  ridge MASE (vs seasonal naive, m=7) = {np.mean(mase_ridge):.3f}")
    assert np.mean(totals["ridge on lags"]) < np.mean(totals["seasonal naive (y[t-7])"]) < np.mean(totals["naive (y[t-1])"])

    print("\n=== Leakage: centered rolling mean + shuffled K-fold ===")
    Xl, yl = leaky_features(y)
    honest, leaky, shuffled = [], [], []
    for (tr, te), (tr2, te2) in zip(walk_forward(len(yl)), KFold(5, shuffle=True, random_state=0).split(Xl)):
        honest.append(mae(yl[te], Ridge().fit(X[:len(yl)][tr], yl[tr]).predict(X[:len(yl)][te])))
        leaky.append(mae(yl[te], Ridge().fit(Xl[tr], yl[tr]).predict(Xl[te])))
        shuffled.append(mae(yl[te2], Ridge().fit(Xl[tr2], yl[tr2]).predict(Xl[te2])))
    print(f"  trailing features, walk-forward  MAE = {np.mean(honest):.2f}  (honest)")
    print(f"  centered mean,     walk-forward  MAE = {np.mean(leaky):.2f}  (looks better: it peeks at y[t])")
    print(f"  centered mean,     shuffled KFold MAE = {np.mean(shuffled):.2f}")
    assert np.mean(leaky) < np.mean(honest)

    print("\n=== Library: TimeSeriesSplit gives the same expanding folds ===")
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5, test_size=60).split(X)):
        print(f"  fold {k}: train rows 0..{tr[-1]}, test rows {te[0]}..{te[-1]}")

    # ---------- figure ----------
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(2, 1, 1)
    days = np.arange(len(y))
    ax.plot(days, y, color=GRAY, lw=0.7, label="synthetic daily demand")
    tr_last, te = list(walk_forward(len(target)))[-1]
    model = Ridge(alpha=1.0).fit(X[tr_last], target[tr_last])
    ax.plot(te + start, model.predict(X[te]), color=BLUE, lw=1.4, label="ridge on lags (last fold)")
    ax.plot(te + start, X[te, 1], color=ORANGE, lw=1, ls="--", label="seasonal naive y[t-7]")
    rolling = np.convolve(y, np.ones(28) / 28, mode="valid")
    ax.plot(np.arange(27, len(y)), rolling, color=PURPLE, lw=1.5, label="28-day trailing mean (trend)")
    ax.set(xlabel="day", ylabel="demand", title="Trend + weekly seasonality, forecast on the last 60 days")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = fig.add_subplot(2, 2, 3)
    for k, (tr, te_) in enumerate(walk_forward(len(target))):
        ax.barh(k, len(tr), color=TEAL, height=0.6)
        ax.barh(k, len(te_), left=te_[0], color=ORANGE, height=0.6)
    ax.set_yticks(range(5), [f"fold {k}" for k in range(5)])
    ax.invert_yaxis()
    ax.set(xlabel="row (time order)", title="Walk-forward: train (teal) always before test (orange)")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(2, 2, 4)
    ax.bar(np.arange(22) - 0.2, r, 0.4, color=BLUE, label="raw series")
    ax.bar(np.arange(22) + 0.2, r_d, 0.4, color=TEAL, label="after diff(7)")
    ax.axhline(2 / np.sqrt(len(y)), color=GRAY, ls=":", lw=1)
    ax.axhline(-2 / np.sqrt(len(y)), color=GRAY, ls=":", lw=1)
    ax.set(xlabel="lag (days)", ylabel="autocorrelation", title="ACF: weekly spikes at 7, 14, 21")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "time-series")
