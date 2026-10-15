"""Features known at forecast time t.

Every feature uses only episode starts at or before t, and an episode's end only when it is
at or before t. An alert still running at t is therefore measured as "running for t - start",
never by its real (future) end. tests/test_leakage.py checks this by recomputing features from
a log truncated at t.
"""
import numpy as np
import pandas as pd

from . import config
from .timeutil import ends_to_ns, to_ns

NS_PER_MIN = 60 * 10**9
WINDOWS_H = {"3h": 3, "6h": 6, "24h": 24, "7d": 168}

FEATURE_COLUMNS = [
    "active_now",
    "current_alert_min",
    "min_since_last_start",
    "min_since_last_end",
    *[f"starts_{w}" for w in WINDOWS_H],
    "active_min_24h",
    "active_min_7d",
    "hour_kyiv",
    "hour_sin",
    "hour_cos",
    "dow_kyiv",
]


def _cumulative_active_ns(x: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Total alert time in (-inf, x] for each x, counting an open alert only up to x.

    Episodes do not overlap, so every episode before the last one started by x has already
    ended by x; only that last one can be partial.
    """
    closed = np.where(e == np.iinfo(np.int64).max, 0, e - s)
    prefix = np.concatenate([[0], np.cumsum(closed)])
    k = np.searchsorted(s, x, side="right")  # episodes started at or before x
    has = k > 0
    last = np.clip(k - 1, 0, None)
    partial = np.where(has, np.minimum(x, e[last]) - s[last], 0)
    return np.where(has, prefix[last] + partial, 0)


def make_features(times: pd.DatetimeIndex, episodes: pd.DataFrame) -> pd.DataFrame:
    t = to_ns(times)
    s = to_ns(episodes["start"])
    e = ends_to_ns(episodes["end"])

    idx = np.searchsorted(s, t, side="right") - 1  # last episode started at or before t
    has = idx >= 0
    i = np.clip(idx, 0, None)
    active = has & (e[i] > t)

    # Last end at or before t: the last episode's end if it is over, otherwise the one before it.
    prev_end = np.where(idx >= 1, e[np.clip(idx - 1, 0, None)], -1)
    last_end = np.where(has & ~active, e[i], prev_end)

    out = {
        "active_now": active.astype(int),
        "current_alert_min": np.where(active, (t - s[i]) / NS_PER_MIN, 0.0),
        "min_since_last_start": np.where(has, (t - s[i]) / NS_PER_MIN, np.nan),
        "min_since_last_end": np.where(last_end >= 0, (t - last_end) / NS_PER_MIN, np.nan),
    }
    for name, hours in WINDOWS_H.items():
        w = hours * 60 * NS_PER_MIN
        out[f"starts_{name}"] = np.searchsorted(s, t, side="right") - np.searchsorted(s, t - w, side="right")

    cum_now = _cumulative_active_ns(t, s, e)
    for name, hours in (("24h", 24), ("7d", 168)):
        w = hours * 60 * NS_PER_MIN
        out[f"active_min_{name}"] = (cum_now - _cumulative_active_ns(t - w, s, e)) / NS_PER_MIN

    local = times.tz_convert(config.TZ)
    out["hour_kyiv"] = local.hour
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    out["dow_kyiv"] = local.dayofweek

    return pd.DataFrame(out, index=times)[FEATURE_COLUMNS]
