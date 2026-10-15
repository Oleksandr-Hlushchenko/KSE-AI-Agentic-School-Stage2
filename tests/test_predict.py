import json
import urllib.error

import pandas as pd
import pytest

from air_alerts import config, fetch
from air_alerts.dataset import build_dataset
from air_alerts.data import build_episodes, parse_events
from air_alerts.fetch import CacheError, check_coverage, load_cache, load_latest, save_cache
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


def history(items: list[dict]) -> bytes:
    return json.dumps({"items": items}).encode()


SNAPSHOT = json.loads(config.RAW_JSON.read_bytes())["items"]  # newest first, as the API sends it
GOOD = config.RAW_JSON.read_bytes()
FETCHED = pd.Timestamp("2026-10-15 17:00", tz="UTC")


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "cache.json"
    save_cache(GOOD, FETCHED, path)
    return path


def without_every_20th(items: list[dict], state: int) -> list[dict]:
    rows = [i for i, x in enumerate(items) if x["state"] == state and x["created_at"] >= "2023"]
    drop = set(rows[::20])
    return [x for i, x in enumerate(items) if i not in drop]


def with_repeated_starts(items: list[dict], every: int) -> list[dict]:
    """Insert a second start one second after every `every`-th start since 2023: all real events
    stay, but pairing breaks (a start while an alert is already open)."""
    out, n = [], 0
    for x in items:  # newest first, so the copy goes before the original
        if x["state"] == 1 and x["created_at"] >= "2023" and (n := n + 1) % every == 0:
            later = (pd.Timestamp(x["created_at"]) + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            out.append({**x, "created_at": later})
        out.append(x)
    return out


STARTS_ONLY = history([x for x in SNAPSHOT if x["state"] == 1])


@pytest.mark.parametrize("bad", [
    payload(("not-a-date", 1)),
    payload(("2026-10-15 10:00", 1)),          # missing seconds
    payload(("2026-10-15 10:00:00", 2)),
    json.dumps({"items": [{"state": 1, "time": "2026-10-15 10:00:00"}]}).encode(),
    json.dumps({"items": []}).encode(),
    json.dumps({"data": []}).encode(),
    b"<html>maintenance</html>",
    # well-formed but unusable for training (review finding 2)
    payload(("2026-10-15 10:30:00", 0)),                                       # one all-clear only
    history([x for x in SNAPSHOT if x["created_at"] >= "2026-01-01"]),         # recent history only
    history(without_every_20th(SNAPSHOT, 1)),                                  # 5% of starts lost
    history(without_every_20th(SNAPSHOT, 0)),                                  # 5% of all-clears lost
    STARTS_ONLY,                                                               # every all-clear lost
    history(with_repeated_starts(SNAPSHOT, every=10)),                         # ~2.5% pairing breaks
], ids=["bad-date", "no-seconds", "bad-state", "no-created_at", "empty", "no-items", "html",
        "end-only", "recent-only", "lost-starts", "lost-all-clears", "starts-only", "broken-pairing"])
def test_invalid_download_never_replaces_good_cache(cache, bad):
    before = cache.read_bytes()
    with pytest.raises(ValueError):
        save_cache(bad, FETCHED, cache)
    assert cache.read_bytes() == before
    assert not list(cache.parent.glob("*.tmp"))
    events, fetched = load_cache(cache)
    assert len(events) == len(SNAPSHOT) and fetched == FETCHED


@pytest.mark.parametrize("bad", [payload(("2026-10-15 10:30:00", 0)), STARTS_ONLY], ids=["end-only", "starts-only"])
def test_failed_download_falls_back_to_cache(cache, monkeypatch, bad):
    """Follow-up review: a starts-only history used to replace the cache and then fail to train."""
    before = cache.read_bytes()
    monkeypatch.setattr(fetch, "download", lambda *a, **k: bad)
    src = load_latest(fetch=True, cache=cache)
    assert cache.read_bytes() == before
    assert not src.fresh and src.retrieved_at == FETCHED and len(src.events) == len(SNAPSHOT)
    assert "download failed" in src.warnings[0]
    forecast_at(src.events, src.retrieved_at)  # the fallback is usable end to end

    def offline(*a, **k):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(fetch, "download", offline)
    src = load_latest(fetch=True, cache=cache)
    assert not src.fresh and len(src.events) == len(SNAPSHOT)


def test_successful_download_replaces_cache(cache, monkeypatch):
    newer = history([{"state": 0, "causes": [], "created_at": "2026-10-15 20:10:00"}, *SNAPSHOT])
    monkeypatch.setattr(fetch, "download", lambda *a, **k: newer)
    src = load_latest(fetch=True, cache=cache)
    assert src.fresh and len(src.events) == len(SNAPSHOT) + 1 and not src.warnings
    assert len(load_cache(cache)[0]) == len(SNAPSHOT) + 1


def corrupt(path, change):
    envelope = json.loads(path.read_text())
    change(envelope)
    path.write_text(json.dumps(envelope))


@pytest.mark.parametrize("change", [
    lambda e: e.update(response=e["response"].replace("19:24:01", "19:24:02")),  # hash mismatch
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


def test_good_history_with_2022_gaps_and_open_alert_is_accepted():
    """The pinned snapshot itself has 53 pairing gaps (all in 2022) and ends with an open alert;
    the coverage check must accept it."""
    events = parse_events(GOOD)
    episodes, anomalies = build_episodes(events)
    assert len(anomalies) == 53 and pd.isna(episodes["end"].iloc[-1])
    check_coverage(events)


def test_missing_cache_uses_pinned_snapshot_quietly(tmp_path):
    src = load_latest(fetch=False, cache=tmp_path / "absent.json")
    assert src.note.startswith("pinned snapshot") and not src.warnings
    assert src.events["t"].is_monotonic_increasing
