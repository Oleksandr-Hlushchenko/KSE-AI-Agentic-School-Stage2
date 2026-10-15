"""Forecast grid, the "new alert starts within the horizon" target, and the time split."""
import numpy as np
import pandas as pd

from . import config
from .timeutil import to_ns


def forecast_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: pd.Timedelta = config.HORIZON,
    freq: str = config.FREQ,
) -> pd.DatetimeIndex:
    """Hourly forecast issue times t (UTC) with start <= t and t + horizon <= end.

    Kyiv is a whole number of hours from UTC, so UTC hours are also Kyiv hours.
    """
    return pd.date_range(start.tz_convert("UTC"), (end - horizon).tz_convert("UTC"), freq=freq)


def label_new_start(
    times: pd.DatetimeIndex,
    episode_starts: pd.Series,
    horizon: pd.Timedelta = config.HORIZON,
) -> np.ndarray:
    """True if at least one alert episode starts in (t, t + horizon].

    A start exactly at t is already known when the forecast is issued, so it belongs to the
    past (it feeds the features), not to the label.
    """
    s = np.sort(to_ns(episode_starts))
    t = to_ns(times)
    h = horizon.value
    return np.searchsorted(s, t + h, side="right") - np.searchsorted(s, t, side="right") > 0


def assign_split(
    times: pd.DatetimeIndex,
    horizon: pd.Timedelta = config.HORIZON,
    val_start: pd.Timestamp = config.VAL_START,
    test_start: pd.Timestamp = config.TEST_START,
) -> np.ndarray:
    """'train' / 'val' / 'test' by issue time, and 'purged' for rows whose label window
    crosses into the next split. Otherwise the last training rows would be labelled with
    alerts that happen inside the validation period."""
    split = np.select([times < val_start, times < test_start], ["train", "val"], "test").astype(object)
    crosses = ((times < val_start) & (times + horizon > val_start)) | (
        (times >= val_start) & (times < test_start) & (times + horizon > test_start)
    )
    split[crosses] = "purged"
    return split
