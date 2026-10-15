import json
import urllib.error

import pandas as pd
import pytest

from air_alerts import config, fetch
from air_alerts.dataset import build_dataset
from air_alerts.fetch import CacheError, load_cache, load_latest, save_cache
from air_alerts.models import DECAY_MODELS
from air_alerts.predict import FINAL_MODEL, describe, forecast_at
from air_alerts.walkforward import walk_forward_predict

# --- forecasting protocol ----------------------------------------------------------------


@pytest.mark.parametrize("at", ["2026-09-20 14:37", "2026-10-03 02:00"])
def test_forecast_ignores_events_after_issue_time(events, at):
    """A replay must give the same answer whether or not the log contains the future."""
    at = config.kyiv(at)
    full = forecast_at(events, at)
    cut = forecast_at(events[events["t"] <= at], at)
    assert full.probability == pytest.approx(cut.probability, abs=1e-12)
    pd.testing.assert_series_equal(full.features, cut.features)


def test_forecast_trains_on_labels_known_at_last_midnight(events):
    at = config.kyiv("2026-09-20 14:37")
    fc = forecast_at(events, at)
    assert fc.refit_at == config.kyiv("2026-09-20")
    assert fc.train_until + config.HORIZON <= fc.refit_at
    assert fc.train_until + config.HORIZON + pd.Timedelta(hours=1) > fc.refit_at
    assert 0 < fc.probability < 1
    assert "P(new alert starts by 2026-09-20 17:37)" in describe(fc)


def test_on_the_hour_forecasts_reproduce_walk_forward_evaluation(events):
    """The live procedure is the evaluated one: same training rows, same features, same model."""
    df = build_dataset(events)
    for day in ("2026-09-18", "2026-10-07"):
        start = config.kyiv(day)
        expected = walk_forward_predict(df, DECAY_MODELS[FINAL_MODEL], start, start + pd.DateOffset(days=1))
        for t in expected.index[[0, 9, 23]]:
            assert forecast_at(events, t).probability == pytest.approx(expected[t], abs=1e-9), t


# --- download cache ----------------------------------------------------------------------


def payload(*events: tuple[str, int]) -> bytes:
    return json.dumps({"items": [{"state": s, "causes": [], "created_at": c} for c, s in events]}).encode()


GOOD = payload(("2026-10-15 10:00:00", 1), ("2026-10-15 10:30:00", 0))
FETCHED = pd.Timestamp("2026-10-15 08:00", tz="UTC")


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "cache.json"
    save_cache(GOOD, FETCHED, path)
    return path


@pytest.mark.parametrize("bad", [
    payload(("not-a-date", 1)),
    payload(("2026-10-15 10:00", 1)),          # missing seconds
    payload(("2026-10-15 10:00:00", 2)),
    json.dumps({"items": [{"state": 1, "time": "2026-10-15 10:00:00"}]}).encode(),
    json.dumps({"items": []}).encode(),
    json.dumps({"data": []}).encode(),
    b"<html>maintenance</html>",
])
def test_invalid_download_never_replaces_good_cache(cache, bad):
    before = cache.read_bytes()
    with pytest.raises(ValueError):
        save_cache(bad, FETCHED, cache)
    assert cache.read_bytes() == before
    assert not list(cache.parent.glob("*.tmp"))
    events, fetched = load_cache(cache)
    assert len(events) == 2 and fetched == FETCHED


def test_failed_download_falls_back_to_cache(cache, monkeypatch):
    monkeypatch.setattr(fetch, "download", lambda *a, **k: payload(("not-a-date", 1)))
    src = load_latest(fetch=True, cache=cache)
    assert not src.fresh and src.retrieved_at == FETCHED and len(src.events) == 2
    assert "download failed" in src.warnings[0]

    def offline(*a, **k):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(fetch, "download", offline)
    src = load_latest(fetch=True, cache=cache)
    assert not src.fresh and len(src.events) == 2


def test_successful_download_replaces_cache(cache, monkeypatch):
    newer = payload(("2026-10-15 10:00:00", 1), ("2026-10-15 10:30:00", 0), ("2026-10-15 11:00:00", 1))
    monkeypatch.setattr(fetch, "download", lambda *a, **k: newer)
    src = load_latest(fetch=True, cache=cache)
    assert src.fresh and len(src.events) == 3 and not src.warnings
    assert len(load_cache(cache)[0]) == 3


def corrupt(path, change):
    envelope = json.loads(path.read_text())
    change(envelope)
    path.write_text(json.dumps(envelope))


@pytest.mark.parametrize("change", [
    lambda e: e.update(response=e["response"].replace("10:30:00", "10:31:00")),  # hash mismatch
    lambda e: e.pop("sha256"),
    lambda e: e.pop("fetched_at_utc"),
    lambda e: e.update(fetched_at_utc="yesterday"),
    lambda e: e.update(fetched_at_utc="2026-10-15T08:00:00"),                  # no time zone
    lambda e: e.update(fetched_at_utc="2999-01-01T00:00:00+00:00"),            # in the future
    lambda e: e.update(version=99),
], ids=["hash", "no-hash", "no-time", "bad-time", "naive-time", "future-time", "version"])
def test_unusable_cache_falls_back_to_pinned_snapshot(cache, change):
    corrupt(cache, change)
    with pytest.raises(CacheError):
        load_cache(cache)
    src = load_latest(fetch=False, cache=cache)
    assert src.note.startswith("pinned snapshot") and "cached data unusable" in src.warnings[0]
    assert src.retrieved_at == config.RAW_RETRIEVED_AT


@pytest.mark.parametrize("content", [b"", b"{not json", b"[]"])
def test_truncated_or_garbled_cache_falls_back(tmp_path, content):
    path = tmp_path / "cache.json"
    path.write_bytes(content)
    src = load_latest(fetch=False, cache=path)
    assert src.note.startswith("pinned snapshot") and src.warnings


def test_missing_cache_uses_pinned_snapshot_quietly(tmp_path):
    src = load_latest(fetch=False, cache=tmp_path / "absent.json")
    assert src.note.startswith("pinned snapshot") and not src.warnings
    assert src.events["t"].is_monotonic_increasing
