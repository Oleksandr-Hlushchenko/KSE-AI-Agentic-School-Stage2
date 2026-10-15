"""Forecast for a single moment: P(a new alert starts in Kyiv within the next 3 hours).

Uses the final model chosen on validation and confirmed on test: logistic regression with
14-day time-decay weights. It follows the evaluated walk-forward protocol exactly: the model is
fit on every label observed by the most recent Kyiv midnight, and only the features use events
up to the forecast time. tests/test_predict.py checks that on-the-hour forecasts reproduce the
walk-forward evaluation. The event log is cut at the forecast time first, so a replay of a past
moment sees only what had been published by then.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .baselines import FlooredRecentRate24h
from .data import build_episodes
from .dataset import build_dataset
from .features import make_features
from .models import DECAY_MODELS

FINAL_MODEL = "logistic+decay14"


@dataclass
class Forecast:
    issued_at: pd.Timestamp          # UTC
    probability: float
    baseline_probability: float      # 24h recent-rate baseline, for reference
    recent_base_rate: float          # share of 3h windows with a new start, last 14 days
    features: pd.Series
    latest_event: pd.Series | None   # newest event at or before issued_at
    train_rows: int
    refit_at: pd.Timestamp           # Kyiv midnight whose observed labels the model is fit on (UTC)
    train_until: pd.Timestamp        # newest training forecast time (label known by refit_at)


def forecast_at(events: pd.DataFrame, at: pd.Timestamp, model: str = FINAL_MODEL) -> Forecast:
    at = at.tz_convert("UTC")
    events = events[events["t"] <= at]  # nothing published after the forecast time
    episodes, _ = build_episodes(events)

    refit_at = at.tz_convert(config.TZ).normalize().tz_convert("UTC")
    history = build_dataset(events, cutoff=refit_at)  # rows with t + horizon <= refit_at
    fitted = DECAY_MODELS[model]().fit(history)

    x = make_features(pd.DatetimeIndex([at]), episodes)
    recent = history[history.index > history.index.max() - pd.Timedelta(days=14)]
    return Forecast(
        issued_at=at,
        probability=float(fitted.predict(x)[0]),
        baseline_probability=float(FlooredRecentRate24h().predict(x)[0]),
        recent_base_rate=float(recent["y"].mean()),
        features=x.iloc[0],
        latest_event=events.iloc[-1] if len(events) else None,
        train_rows=len(history),
        refit_at=refit_at,
        train_until=history.index.max(),
    )


def describe(fc: Forecast) -> str:
    """Human-readable summary in Kyiv time."""
    def kyiv(ts: pd.Timestamp) -> str:
        return ts.tz_convert(config.TZ).strftime("%Y-%m-%d %H:%M")

    f = fc.features
    until = fc.issued_at + config.HORIZON
    if f["active_now"]:
        status = f"ALERT ACTIVE for {f['current_alert_min']:.0f} min"
    elif np.isnan(f["min_since_last_end"]):
        status = "no alert"
    else:
        status = f"no alert; last all-clear {f['min_since_last_end']:.0f} min ago"
    if fc.latest_event is not None:
        kind = "start" if fc.latest_event["state"] == 1 else "all-clear"
        latest = f"{kyiv(fc.latest_event['t'])} ({kind})"
    else:
        latest = "none"

    return "\n".join([
        f"Kyiv City air-alert forecast, issued {kyiv(fc.issued_at)} Kyiv time",
        "",
        f"  P(new alert starts by {kyiv(until)}):  {fc.probability:.0%}",
        "",
        f"  Status now:            {status}",
        f"  Alert starts:          {f['starts_3h']:.0f} in last 3h, {f['starts_24h']:.0f} in last 24h, {f['starts_7d']:.0f} in last 7 days",
        f"  Latest event in data:  {latest}",
        "",
        "  For comparison:",
        f"    24h-rate baseline:                      {fc.baseline_probability:.0%}",
        f"    3h windows with a new start, last 14d:  {fc.recent_base_rate:.0%}",
        "",
        f"  Model: logistic regression, 14-day time-decay, fit on {fc.train_rows} hourly forecasts whose",
        f"  outcome was known at {kyiv(fc.refit_at)} (daily refit at Kyiv midnight, as evaluated).",
        "",
        "  Historical evaluation, 17 Sep - 14 Oct 2026 (daily refit at midnight, forecasts issued on",
        "  the hour): Brier skill +0.04 vs the period's base rate, ROC-AUC 0.64. The same forecasts",
        "  moved to random minutes scored alike (+0.05, 0.65; scripts/check_minute_offsets.py).",
        "  Mostly right on the overall level, weak at telling individual hours apart.",
        "  Not an official warning.",
    ])
