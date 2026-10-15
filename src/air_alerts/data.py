"""Loading the Kyiv Digital event log and turning it into alert episodes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_raw(path: Path = config.RAW_JSON, sha256: str | None = config.RAW_SHA256) -> pd.DataFrame:
    raw = Path(path).read_bytes()
    if sha256 is not None:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sha256:
            raise ValueError(f"{path}: sha256 {digest} does not match the pinned snapshot {sha256}")
    return pd.DataFrame(json.loads(raw)["items"], columns=["created_at", "state", "causes"])


def kyiv_local_to_utc(local: pd.Series) -> pd.Series:
    """Convert offset-less Kyiv wall-clock strings to UTC.

    Spring-forward gaps cannot occur in real data, but are shifted forward to be safe.
    In the repeated autumn hour the first occurrence (summer time) is assumed unless that
    would put the event before the previous one; `local` must be sorted.
    """
    naive = pd.to_datetime(local, format="%Y-%m-%d %H:%M:%S")
    utc = (
        naive.dt.tz_localize(config.TZ, ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    for i in np.flatnonzero(utc.isna().to_numpy()):
        early = naive.iloc[i].tz_localize(config.TZ, ambiguous=True).tz_convert("UTC")
        late = naive.iloc[i].tz_localize(config.TZ, ambiguous=False).tz_convert("UTC")
        prev = utc.iloc[i - 1] if i > 0 else pd.NaT
        utc.iloc[i] = late if pd.notna(prev) and early < prev else early
    return utc


def load_events(
    path: Path = config.RAW_JSON,
    sha256: str | None = config.RAW_SHA256,
    cutoff: pd.Timestamp = config.CUTOFF,
) -> pd.DataFrame:
    """Start (state=1) / end (state=0) events with UTC time `t`, sorted, strictly before `cutoff`.

    `causes` is kept for after-the-fact analysis only. It must never become a model input:
    start records carry threat types that were most likely added later in the alert.
    """
    df = load_raw(path, sha256)
    # "YYYY-mm-dd HH:MM:SS" sorts chronologically as a string.
    df = df.sort_values("created_at", kind="stable").reset_index(drop=True)
    df["t"] = kyiv_local_to_utc(df["created_at"])
    df = df[df["t"] < cutoff.tz_convert("UTC")].reset_index(drop=True)
    return df[["t", "state", "causes", "created_at"]]


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
