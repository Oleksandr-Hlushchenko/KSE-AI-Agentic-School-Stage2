"""Forecast whether a new air-raid alert starts in Kyiv City within the next 3 hours.

    python scripts/predict.py                          # download fresh data, forecast for now
    python scripts/predict.py --offline                # use the last downloaded data
    python scripts/predict.py --at "2026-10-10 18:00"  # replay a past moment (Kyiv time)

Fresh data is cached in data/live/ (not in git) only after it parses completely. Offline and
replay modes use that cache if it is intact, otherwise the pinned snapshot in data/raw/. Without
fresh data the forecast is issued as of the moment the data was retrieved, not "now": the alert
status after that moment is unknown.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_alerts import config  # noqa: E402
from air_alerts.fetch import load_latest  # noqa: E402
from air_alerts.predict import describe, forecast_at  # noqa: E402


def kyiv(ts: pd.Timestamp) -> str:
    return ts.tz_convert(config.TZ).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="do not download; use cached or pinned data")
    parser.add_argument("--at", help='replay a past moment, Kyiv time, e.g. "2026-10-10 18:00"')
    args = parser.parse_args()

    source = load_latest(fetch=not (args.offline or args.at))
    for warning in source.warnings:
        print(f"WARNING: {warning}\n", file=sys.stderr)
    events, retrieved_at = source.events, source.retrieved_at
    if args.at:
        at = pd.Timestamp(args.at, tz=config.TZ)
        if at > retrieved_at:
            sys.exit(f"--at {args.at} is after the data was retrieved ({kyiv(retrieved_at)} Kyiv); "
                     "run without --offline/--at to download fresh data")
    elif source.fresh:
        at = pd.Timestamp.now(tz="UTC")
    else:
        at = retrieved_at
        age_h = (pd.Timestamp.now(tz="UTC") - retrieved_at) / pd.Timedelta(hours=1)
        print(f"WARNING: no fresh data. Forecast is issued as of {kyiv(at)} Kyiv, when the data was\n"
              f"retrieved ({age_h:.1f} h ago). The alert status after that moment is unknown.\n")
    if at < config.DATA_START + pd.Timedelta(days=30):
        sys.exit("--at must be at least 30 days after 2023-01-01 so the model has history to learn from")

    print(describe(forecast_at(events, at)))
    print(f"\n  Data: {source.note}")


if __name__ == "__main__":
    main()
