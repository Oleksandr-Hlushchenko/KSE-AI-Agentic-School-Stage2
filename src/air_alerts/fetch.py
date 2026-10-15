"""Download the current Kyiv Digital alert history (public endpoint, no token needed) and keep
the last good copy.

The cache is a single JSON file holding the raw response together with its retrieval time and
sha256, replaced atomically. The response is fully parsed and checked for coverage before it is
written, so a bad or partial download never overwrites a usable cache, and the payload and its
metadata can't get out of step.
"""
import hashlib
import json
import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import certifi
import pandas as pd

from . import config
from .data import build_episodes, load_events, parse_events
from .labels import forecast_times, label_new_start

CACHE_VERSION = 1
# Allowed clock difference when checking that a cache wasn't written "in the future".
CLOCK_SLACK = pd.Timedelta(minutes=5)


# Features look back up to 7 days from the first training row.
FEATURE_LOOKBACK = pd.Timedelta(days=7)
# Share of the pinned snapshot's starts and all-clears a new history must still contain.
MIN_SNAPSHOT_OVERLAP = 0.99
# Allowed share of events since the training start that break start/all-clear pairing.
# The pinned snapshot has none; its 53 known gaps are all in 2022, before the training start.
MAX_ANOMALY_SHARE = 0.01


class CacheError(Exception):
    """The cached download exists but can't be trusted."""


def check_coverage(events: pd.DataFrame) -> None:
    """Raise ValueError unless `events` is a full history the model can train on.

    A missing stretch of history is not "no alerts": it would turn into negative labels, and
    missing all-clears would merge many alerts into one that never ends. So the log must:
    - reach back before the training start (minus the 7-day feature window);
    - still contain the pinned snapshot's alert starts *and* all-clears over the snapshot's
      period (the API history only grows; two downloads 3.5 hours apart matched exactly);
    - pair starts with all-clears since the training start, apart from a few gaps (a final
      open alert is fine, and the known 2022 gaps are before the training start);
    - give both outcomes, "new alert within 3h" and "none", as training labels.
    """
    earliest = config.DATA_START - FEATURE_LOOKBACK
    if events.empty or events["t"].min() > earliest:
        first = "nothing" if events.empty else events["t"].min()
        raise ValueError(f"history starts at {first}, needs events from before {earliest}")

    pinned = load_events(cutoff=None)
    in_pinned = (pinned["t"] >= config.DATA_START) & (pinned["t"] < config.RAW_RETRIEVED_AT)
    in_events = (events["t"] >= config.DATA_START) & (events["t"] < config.RAW_RETRIEVED_AT)
    for state, name in ((1, "alert starts"), (0, "all-clears")):
        expected = int((pinned.loc[in_pinned, "state"] == state).sum())
        found = int((events.loc[in_events, "state"] == state).sum())
        if found < MIN_SNAPSHOT_OVERLAP * expected:
            raise ValueError(f"history has {found} {name} in {config.DATA_START:%Y-%m-%d}.."
                             f"{config.RAW_RETRIEVED_AT:%Y-%m-%d}, the pinned snapshot has {expected}")

    episodes, anomalies = build_episodes(events)
    recent_events = int((events["t"] >= config.DATA_START).sum())
    recent_anomalies = int((anomalies["t"] >= config.DATA_START).sum()) if len(anomalies) else 0
    if recent_anomalies > MAX_ANOMALY_SHARE * recent_events:
        raise ValueError(f"{recent_anomalies} of {recent_events} events since {config.DATA_START:%Y-%m-%d} "
                         "break start/all-clear pairing")

    labels = label_new_start(forecast_times(config.DATA_START, events["t"].max()), episodes["start"])
    if labels.all() or not labels.any():
        raise ValueError("history yields only one outcome class for training")


def download(url: str = config.HISTORY_URL, timeout: float = 30) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "kse-air-alerts/0.1"})
    # certifi's CA bundle: python.org builds of Python on macOS ship without root certificates.
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def save_cache(raw: bytes, fetched_at: pd.Timestamp, dest: Path = config.LIVE_CACHE, url: str = config.HISTORY_URL) -> pd.DataFrame:
    """Validate `raw`, then atomically replace the cache. Returns the parsed events."""
    events = parse_events(raw)  # raises before anything is written
    check_coverage(events)
    envelope = {
        "version": CACHE_VERSION,
        "source_url": url,
        "fetched_at_utc": fetched_at.tz_convert("UTC").isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "response": raw.decode("utf-8"),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        tmp.write_text(json.dumps(envelope))
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return events


def fetch_history(dest: Path = config.LIVE_CACHE, url: str = config.HISTORY_URL) -> tuple[pd.DataFrame, pd.Timestamp]:
    raw = download(url)
    fetched_at = pd.Timestamp.now(tz="UTC")
    return save_cache(raw, fetched_at, dest, url), fetched_at


def load_cache(path: Path = config.LIVE_CACHE) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Events and retrieval time from the cache. Raises CacheError if any part is unusable."""
    try:
        envelope = json.loads(path.read_bytes())
        if envelope.get("version") != CACHE_VERSION:
            raise ValueError(f"unknown cache version {envelope.get('version')!r}")
        raw = envelope["response"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
            raise ValueError("sha256 does not match the stored response")
        fetched_at = pd.Timestamp(envelope["fetched_at_utc"])
        if fetched_at.tzinfo is None or fetched_at > pd.Timestamp.now(tz="UTC") + CLOCK_SLACK:
            raise ValueError(f"implausible retrieval time {envelope['fetched_at_utc']!r}")
        events = parse_events(raw)
        check_coverage(events)
        return events, fetched_at.tz_convert("UTC")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise CacheError(f"{path}: {exc}") from exc


@dataclass
class Source:
    events: pd.DataFrame
    retrieved_at: pd.Timestamp  # UTC; the alert status after this moment is unknown
    note: str
    fresh: bool
    warnings: list[str] = field(default_factory=list)


def load_latest(fetch: bool, cache: Path = config.LIVE_CACHE) -> Source:
    """Fresh download if asked and possible, else the cache if usable, else the pinned snapshot."""
    warnings = []
    if fetch:
        try:
            events, at = fetch_history(cache)
            return Source(events, at, f"downloaded just now ({len(events)} events)", True, warnings)
        except Exception as exc:  # network, HTTP, TLS or payload problems: fall back below
            warnings.append(f"download failed ({exc}); the last good copy was kept")
    if cache.exists():
        try:
            events, at = load_cache(cache)
            local = at.tz_convert(config.TZ).strftime("%Y-%m-%d %H:%M")
            return Source(events, at, f"cached download from {local} Kyiv", False, warnings)
        except CacheError as exc:
            warnings.append(f"cached data unusable ({exc}); using the pinned snapshot")
    events = load_events(cutoff=None)
    note = f"pinned snapshot {config.RAW_JSON.relative_to(config.ROOT)}"
    return Source(events, config.RAW_RETRIEVED_AT.tz_convert("UTC"), note, False, warnings)
