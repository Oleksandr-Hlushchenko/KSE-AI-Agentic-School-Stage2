import numpy as np
import pandas as pd

from air_alerts import config
from air_alerts.labels import assign_split, forecast_times, label_new_start
from air_alerts.timeutil import to_ns


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_label_window_is_open_closed():
    t = pd.DatetimeIndex([ts("2025-01-01 12:00")])
    h = pd.Timedelta(hours=3)
    cases = {
        "2025-01-01 12:00:00": False,  # exactly at t: already known, not a future start
        "2025-01-01 12:00:01": True,
        "2025-01-01 15:00:00": True,   # exactly at t + 3h: included
        "2025-01-01 15:00:01": False,
        "2025-01-01 11:59:59": False,
    }
    for start, expected in cases.items():
        assert label_new_start(t, pd.Series([ts(start)]), h)[0] == expected, start


def test_grid_label_windows_end_by_cutoff():
    times = forecast_times(config.DATA_START, config.CUTOFF)
    assert times[0] == config.DATA_START
    assert times[-1] + config.HORIZON <= config.CUTOFF
    assert times[-1] + config.HORIZON + pd.Timedelta(hours=1) > config.CUTOFF
    assert (np.diff(to_ns(times)) == 3600 * 10**9).all()


def test_split_purges_windows_crossing_boundaries():
    times = forecast_times(config.DATA_START, config.CUTOFF)
    split = assign_split(times)
    end = times + config.HORIZON
    assert (end[split == "train"] <= config.VAL_START).all()
    assert (times[split == "val"] >= config.VAL_START).all()
    assert (end[split == "val"] <= config.TEST_START).all()
    assert (times[split == "test"] >= config.TEST_START).all()
    # horizon 3h, hourly grid -> 2 purged rows before each boundary
    assert (split == "purged").sum() == 4
    # splits are contiguous blocks in time order
    order = {"train": 0, "val": 1, "test": 2}
    ranks = [order[s] for s in split if s != "purged"]
    assert ranks == sorted(ranks)
