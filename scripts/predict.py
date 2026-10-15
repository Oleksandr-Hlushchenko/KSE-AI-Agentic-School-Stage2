"""Forecast whether a new air-raid alert starts in Kyiv City within the next 3 hours.

    python scripts/predict.py                          # download fresh data, forecast for now
    python scripts/predict.py --offline                # use the last downloaded data
    python scripts/predict.py --at "2026-10-10 18:00"  # replay a past moment (Kyiv time)
    python scripts/predict.py --details                # add model inputs, training data, metrics

The default output is a short message in Ukrainian; --details appends the technical summary
(in English) and any technical warnings.

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
from air_alerts.predict import describe_details, describe_simple, duration_uk, forecast_at  # noqa: E402


def kyiv_uk(ts: pd.Timestamp) -> str:
    return ts.tz_convert(config.TZ).strftime("%d.%m.%Y, %H:%M")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="do not download; use cached or pinned data")
    parser.add_argument("--at", help='replay a past moment, Kyiv time, e.g. "2026-10-10 18:00"')
    parser.add_argument("--details", action="store_true", help="also print model inputs, training data and metrics")
    args = parser.parse_args()

    fetch = not (args.offline or args.at)
    source = load_latest(fetch=fetch)
    if args.details:
        for warning in source.warnings:
            print(f"WARNING: {warning}\n", file=sys.stderr)

    notes = []
    if args.at:
        try:
            at = pd.Timestamp(args.at, tz=config.TZ)
        except ValueError:
            sys.exit(f'Не вдалося розпізнати час "{args.at}". Приклад: --at "2026-10-10 18:00"')
        if at > source.retrieved_at:
            sys.exit(f"Дані отримано {kyiv_uk(source.retrieved_at)}, тож прогноз на пізніший момент "
                     "неможливий. Запустіть без --offline і --at, щоб завантажити свіжі дані.")
        notes.append("Відтворення минулого моменту: використано лише дані, відомі на той час.")
    elif source.fresh:
        at = pd.Timestamp.now(tz="UTC")
    else:
        at = source.retrieved_at
        age = duration_uk((pd.Timestamp.now(tz="UTC") - at) / pd.Timedelta(minutes=1))
        reason = "не вдалося завантажити свіжі дані." if fetch else "режим без завантаження (--offline)."
        notes.append(f"Увага: {reason} Прогноз зроблено станом на {kyiv_uk(at)}, коли дані було\n"
                     f"отримано ({age} тому). Що відбувалося після цього, невідомо.")
    if at < config.DATA_START + pd.Timedelta(days=30):
        sys.exit("Замало історії для прогнозу: потрібен момент не раніше 31.01.2023.")

    forecast = forecast_at(source.events, at)
    for note in notes:
        print(note + "\n")
    print(describe_simple(forecast))
    if args.details:
        print("\n" + "-" * 72)
        print(describe_details(forecast))
        print(f"\n  Data: {source.note}")


if __name__ == "__main__":
    main()
