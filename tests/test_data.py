import pandas as pd
import pytest

from air_alerts import config
from air_alerts.data import build_episodes, kyiv_local_to_utc, load_raw

from .conftest import make_events


def test_snapshot_hash_is_enforced(tmp_path):
    bad = tmp_path / "history.json"
    bad.write_text('{"items": []}')
    with pytest.raises(ValueError, match="sha256"):
        load_raw(bad)


@pytest.mark.parametrize("local, utc", [
    ("2025-01-15 12:00:00", "2025-01-15 10:00:00"),  # winter, UTC+2
    ("2025-07-15 12:00:00", "2025-07-15 09:00:00"),  # summer, UTC+3
    ("2025-03-30 03:30:00", "2025-03-30 01:00:00"),  # nonexistent spring hour -> shifted to 04:00 EEST
])
def test_kyiv_local_to_utc(local, utc):
    assert kyiv_local_to_utc(pd.Series([local])).iloc[0] == pd.Timestamp(utc, tz="UTC")


def test_repeated_autumn_hour_keeps_order():
    # 2025-10-26 03:00-04:00 happens twice in Kyiv. 03:40 then 03:10 in a sorted log is impossible,
    # but 03:10 (first pass, EEST) followed by 03:40 must stay ordered either way.
    got = kyiv_local_to_utc(pd.Series(["2025-10-26 02:50:00", "2025-10-26 03:10:00", "2025-10-26 03:40:00"]))
    assert got.is_monotonic_increasing
    assert got.iloc[1] == pd.Timestamp("2025-10-26 00:10:00", tz="UTC")


def test_real_log_sorted_and_before_cutoff(events):
    assert events["t"].is_monotonic_increasing
    assert events["t"].max() < config.CUTOFF
    assert set(events["state"].unique()) == {0, 1}


def test_log_is_clean_since_data_start(events):
    """Starts and ends strictly alternate from DATA_START on; the known gaps are all in 2022."""
    _, anomalies = build_episodes(events)
    assert len(anomalies) == 53
    assert (anomalies["t"] < config.DATA_START).all()


def test_episodes_do_not_overlap(episodes):
    closed = episodes.dropna(subset=["end"])
    assert (closed["end"] > closed["start"]).all()
    assert (episodes["start"].iloc[1:].to_numpy() >= episodes["end"].iloc[:-1].to_numpy()).all()


def test_build_episodes_handles_open_and_missing_events():
    ev = make_events([
        ("2025-01-01 00:00", 1), ("2025-01-01 00:30", 1),  # repeated start
        ("2025-01-01 01:00", 0), ("2025-01-01 01:10", 0),  # repeated end
        ("2025-01-01 02:00", 1),                           # still open
    ])
    episodes, anomalies = build_episodes(ev)
    assert list(anomalies["kind"]) == ["start_while_active", "end_while_clear"]
    assert len(episodes) == 2
    assert pd.isna(episodes["end"].iloc[-1])
