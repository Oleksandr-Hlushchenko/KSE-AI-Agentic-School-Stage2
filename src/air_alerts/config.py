"""Project-wide constants: data snapshot, time zone, forecast horizon and the time split.

All split boundaries are Kyiv wall-clock midnights; internally everything is UTC.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Raw snapshot of https://kyiv.digital/open-api/air-alert/history retrieved 2026-10-15.
# The hash pins the exact bytes we analysed; loading fails if the file changes.
RAW_JSON = ROOT / "data" / "raw" / "kyiv_digital" / "history_2026-10-15.json"
RAW_SHA256 = "417e4024d258163c9ead52f80542d2566da0ea0a700bba70d933bbd616df1b3d"
RAW_RETRIEVED_AT = pd.Timestamp("2026-10-15 19:59", tz="Europe/Kyiv")  # file time of the download

# Live data for scripts/predict.py. Kept apart from the pinned snapshot (and out of git)
# so that evaluation stays reproducible.
HISTORY_URL = "https://kyiv.digital/open-api/air-alert/history"
LIVE_CACHE = ROOT / "data" / "live" / "history_cache.json"

# created_at in the API has no offset; it is Kyiv local time with DST (see DECISIONS.md).
TZ = "Europe/Kyiv"

HORIZON = pd.Timedelta(hours=3)
FREQ = "h"


def kyiv(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=TZ)


# 2022 has repeated starts/ends (missing events); from 2023 the event log is clean.
DATA_START = kyiv("2023-01-01")
VAL_START = kyiv("2026-08-24")   # escalation begins here; validation must see it
TEST_START = kyiv("2026-09-17")  # last 4 weeks, evaluated once at the end
# Events at or after the cutoff are dropped, and every label window must end by it.
# The snapshot itself runs to 2026-10-15 19:24 with an alert still open.
CUTOFF = kyiv("2026-10-15")
