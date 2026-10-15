"""Leakage tests: nothing after the forecast time may influence features or fitted baselines."""
import numpy as np
import pandas as pd
import pytest

from air_alerts import config
from air_alerts.baselines import BASELINES
from air_alerts.data import build_episodes
from air_alerts.dataset import build_dataset
from air_alerts.features import FEATURE_COLUMNS, make_features
from air_alerts.labels import forecast_times

from .conftest import make_events


def _sample_times(n: int, seed: int = 0) -> pd.DatetimeIndex:
    times = forecast_times(config.DATA_START, config.CUTOFF)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(times), size=n, replace=False)
    # Always include the most recent period, where alerts are densest.
    recent = np.flatnonzero(times >= config.VAL_START)[:: max(1, len(times) // 2000)]
    return times[np.unique(np.concatenate([picked, recent]))]


def test_features_identical_when_log_is_truncated_at_t(events):
    """The core leakage test: rebuild everything from only the events published by t."""
    full_episodes, _ = build_episodes(events)
    for t in _sample_times(300):
        idx = pd.DatetimeIndex([t])
        full = make_features(idx, full_episodes)
        truncated_episodes, _ = build_episodes(events[events["t"] <= t])
        truncated = make_features(idx, truncated_episodes)
        pd.testing.assert_frame_equal(full, truncated, check_exact=False, rtol=1e-12, obj=str(t))


def test_future_events_do_not_change_features():
    past = [("2025-01-01 10:00", 1), ("2025-01-01 10:40", 0), ("2025-01-01 11:30", 1)]
    t = pd.DatetimeIndex([pd.Timestamp("2025-01-01 12:00", tz="UTC")])
    base = make_features(t, build_episodes(make_events(past))[0])
    for future in (
        [("2025-01-01 12:05", 0)],                                  # open alert ends after t
        [("2025-01-01 13:00", 0), ("2025-01-01 13:10", 1)],
        [("2025-01-01 12:00:01", 0), ("2025-01-01 12:30", 1), ("2025-01-02 00:00", 0)],
    ):
        changed = make_features(t, build_episodes(make_events(past + future))[0])
        pd.testing.assert_frame_equal(base, changed)


def test_feature_values_on_hand_example():
    ev = make_events([("2025-01-01 10:00", 1), ("2025-01-01 10:40", 0), ("2025-01-01 11:30", 1)])
    t = pd.DatetimeIndex([pd.Timestamp("2025-01-01 12:00", tz="UTC")])
    f = make_features(t, build_episodes(ev)[0]).iloc[0]
    assert f["active_now"] == 1
    assert f["current_alert_min"] == 30
    assert f["min_since_last_start"] == 30
    assert f["min_since_last_end"] == 80
    assert f["starts_3h"] == 2
    assert f["active_min_24h"] == 40 + 30
    assert f["hour_kyiv"] == 14  # 12:00 UTC in winter


def test_causes_is_not_a_feature():
    assert not any("cause" in c for c in FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def dataset(events):
    return build_dataset(events)


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_ignore_labels_outside_training_rows(dataset, name):
    """Fitting on train must not depend on val/test labels: scramble them and predict again."""
    train = dataset[dataset["split"] == "train"]
    scrambled = dataset.copy()
    other = scrambled["split"] != "train"
    scrambled.loc[other, "y"] = 1 - scrambled.loc[other, "y"]

    p1 = BASELINES[name]().fit(train).predict(dataset)
    p2 = BASELINES[name]().fit(scrambled[scrambled["split"] == "train"]).predict(scrambled)
    np.testing.assert_allclose(p1, p2)
