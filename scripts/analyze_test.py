"""Uncertainty and breakdowns for the one-time test result (descriptive; no model is changed).

    python scripts/analyze_test.py

- Brier improvement between forecasters with a day-block bootstrap: whole Kyiv days are
  resampled with replacement (28 days, 2000 replicates, seed 0), because hours within a day are
  strongly correlated. The interval describes uncertainty over *these* 28 days; it says nothing
  about how the model would do under a different regime.
- Weekly Brier skill, and observed vs predicted rates by recent activity.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_alerts import config  # noqa: E402
from air_alerts.baselines import FlooredRecentRate24h  # noqa: E402
from air_alerts.dataset import build_dataset  # noqa: E402
from air_alerts.models import DECAY_MODELS, MODELS  # noqa: E402
from air_alerts.walkforward import walk_forward_predict  # noqa: E402

SEED, REPS = 0, 2000


def bootstrap_brier_gain(y, p_ref, p_new, days, rng):
    """Brier(ref) - Brier(new): positive means `new` is better. Day-block bootstrap CI."""
    by_day = [np.flatnonzero(days == d) for d in np.unique(days)]
    loss_ref, loss_new = (p_ref - y) ** 2, (p_new - y) ** 2
    reps = np.empty(REPS)
    for r in range(REPS):
        idx = np.concatenate([by_day[k] for k in rng.integers(0, len(by_day), len(by_day))])
        reps[r] = loss_ref[idx].mean() - loss_new[idx].mean()
    return loss_ref.mean() - loss_new.mean(), np.percentile(reps, [2.5, 97.5]), (reps > 0).mean()


def bss(y, p):
    return 1 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean()))


def main() -> None:
    df = build_dataset()
    test = df[df["split"] == "test"]
    y = test["y"].to_numpy()
    preds = {
        "baseline_24h": FlooredRecentRate24h().predict(test),
        "lightgbm_wf": walk_forward_predict(df, MODELS["lightgbm"], config.TEST_START, config.CUTOFF).reindex(test.index).to_numpy(),
        "logistic_decay14_wf": walk_forward_predict(df, DECAY_MODELS["logistic+decay14"], config.TEST_START, config.CUTOFF).reindex(test.index).to_numpy(),
    }
    local = test.index.tz_convert(config.TZ)
    days = np.asarray(local.normalize())

    rng = np.random.default_rng(SEED)
    print(f"Brier gain on test (positive = second is better); day-block bootstrap, "
          f"{len(np.unique(days))} days, {REPS} reps, seed {SEED}")
    for ref, new in [("baseline_24h", "logistic_decay14_wf"), ("baseline_24h", "lightgbm_wf"),
                     ("lightgbm_wf", "logistic_decay14_wf")]:
        gain, (lo, hi), share = bootstrap_brier_gain(y, preds[ref], preds[new], days, rng)
        print(f"  {new} vs {ref}: {gain:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  share of reps > 0: {share:.2f}")

    week = np.asarray((local - config.TEST_START).days // 7)
    print("\nWeekly Brier skill vs that week's own base rate")
    for w in np.unique(week):
        m = week == w
        cells = "  ".join(f"{k} {bss(y[m], p[m]):+.3f}" for k, p in preds.items())
        print(f"  week {w + 1}: base rate {y[m].mean():.2f}  {cells}")

    bins = pd.cut(test["starts_24h"], [-1, 2, 4, 6, 9, 12, 40])
    table = pd.DataFrame({"y": y, **preds}, index=test.index).groupby(bins, observed=True)
    print("\nObserved vs mean predicted, by alert starts in the last 24h")
    print(table.agg(["mean"]).droplevel(1, axis=1).assign(hours=table.size()).round(2).to_string())


if __name__ == "__main__":
    main()
