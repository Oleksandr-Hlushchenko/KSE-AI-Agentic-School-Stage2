"""Loading the Kyiv Digital event log and turning it into alert episodes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_raw(path: Path = config.RAW_JSON, sha256: str | None = config.RAW_SHA256) -> bytes:
    raw = Path(path).read_bytes()
    if sha256 is not None:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sha256:
            raise ValueError(f"{path}: sha256 {digest} does not match the pinned snapshot {sha256}")
    return raw


def localize_kyiv(local: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert offset-less Kyiv wall-clock strings, given in chronological order, to UTC.

    When clocks go back, 03:00-04:00 happens twice and the string alone can't say which pass an
    event belongs to; the order of events can. An event in that hour is placed in the first
    (summer-time) pass unless that would put it before the previous event: so a start at 03:50
    followed by an end at 03:10 becomes 00:50 -> 01:10 UTC. Where both passes fit between the
    neighbouring events, the order can't decide; the first pass is assumed and the event is
    flagged in the second return value. Spring-forward gaps can't occur in real data and are
    shifted forward to be safe.
    """
    naive = pd.to_datetime(local, format="%Y-%m-%d %H:%M:%S")
    utc = (
        naive.dt.tz_localize(config.TZ, ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    ambiguous = np.flatnonzero(utc.isna().to_numpy())
    options = {}
    for i in ambiguous:
        early = naive.iloc[i].tz_localize(config.TZ, ambiguous=True).tz_convert("UTC")
        late = naive.iloc[i].tz_localize(config.TZ, ambiguous=False).tz_convert("UTC")
        prev = utc.iloc[i - 1] if i > 0 else pd.NaT
        utc.iloc[i] = late if pd.notna(prev) and early < prev else early
        options[i] = (early, late)

    assumed = pd.Series(False, index=local.index)
    for i, (early, late) in options.items():
        prev = utc.iloc[i - 1] if i > 0 else pd.NaT
        nxt = utc.iloc[i + 1] if i + 1 < len(utc) else pd.NaT

        def fits(x):
            return (pd.isna(prev) or x >= prev) and (pd.isna(nxt) or x <= nxt)
        assumed.iloc[i] = bool(fits(early) and fits(late))
    return utc, assumed


def kyiv_local_to_utc(local: pd.Series) -> pd.Series:
    return localize_kyiv(local)[0]


def parse_events(raw: bytes, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Start (state=1) / end (state=0) events with UTC time `t`, in time order, strictly before
    `cutoff` (no cutoff if None, for live data). Raises ValueError on anything the pipeline can't
    use. `dst_assumed` marks events whose pass through the repeated autumn hour was assumed (see
    localize_kyiv).

    `causes` is kept for after-the-fact analysis only. It must never become a model input:
    the API rewrites a start record's threat types later in the alert (observed: the 19:24 start on
    2026-10-15 read ["drone"] at 20:00 and ["drone", "missile"] at 23:30).
    """
    try:
        items = json.loads(raw)["items"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"not an alert-history response: {exc!r}") from exc
    if not isinstance(items, list) or not items:
        raise ValueError("alert history has no events")
    for item in items:
        if not isinstance(item, dict) or not {"state", "created_at"} <= set(item):
            raise ValueError(f"event without state/created_at: {item!r}")
        if item["state"] not in (0, 1) or isinstance(item["state"], bool) or not isinstance(item["created_at"], str):
            raise ValueError(f"malformed event: {item!r}")

    df = pd.DataFrame(items).reindex(columns=["created_at", "state", "causes"])
    # The API lists events newest first (all 5,386 events of the snapshot, no exception). That
    # order, not sorting the clock strings, is what tells the two passes through the repeated
    # autumn hour apart, so it is kept, reversed. The order can't be inferred from the strings
    # themselves (two events inside the repeated hour look the same either way), so any other
    # order fails the time-order check below instead of being guessed.
    df = df.iloc[::-1].reset_index(drop=True)
    df["t"], df["dst_assumed"] = localize_kyiv(df["created_at"])  # strict format: raises on bad strings
    if df["t"].isna().any():
        raise ValueError("unparseable created_at")
    if not df["t"].is_monotonic_increasing:
        raise ValueError("events are not in a consistent time order")
    if cutoff is not None:
        df = df[df["t"] < cutoff.tz_convert("UTC")].reset_index(drop=True)
    return df[["t", "state", "causes", "created_at", "dst_assumed"]]


def load_events(
    path: Path = config.RAW_JSON,
    sha256: str | None = config.RAW_SHA256,
    cutoff: pd.Timestamp | None = config.CUTOFF,
) -> pd.DataFrame:
    return parse_events(load_raw(path, sha256), cutoff)


def build_episodes(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair starts with ends into non-overlapping episodes.

    A start while an alert is already open, or an end while none is open, is a missing event
    in the log (all 53 cases are in 2022). Those are reported as anomalies and the episode
    stays as it was. An alert open at the end of `events` gets end = NaT.
    """
    starts, ends, anomalies = [], [], []
    open_start = None
    for t, state in zip(events["t"], events["state"]):
        if state == 1:
            if open_start is None:
                open_start = t
            else:
                anomalies.append((t, "start_while_active"))
        elif open_start is not None:
            starts.append(open_start)
            ends.append(t)
            open_start = None
        else:
            anomalies.append((t, "end_while_clear"))
    if open_start is not None:
        starts.append(open_start)
        ends.append(pd.NaT)

    episodes = pd.DataFrame({
        "start": pd.to_datetime(pd.Series(starts, dtype=object), utc=True),
        "end": pd.to_datetime(pd.Series(ends, dtype=object), utc=True),
    })
    return episodes, pd.DataFrame(anomalies, columns=["t", "kind"])
