"""Score the baselines on validation (default) or, once, on test.

    python scripts/run_baselines.py            # fit on train, score on val
    python scripts/run_baselines.py --test     # fit on train+val, score on test

The test period is meant to be looked at once, after all choices are made on validation.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_alerts import config  # noqa: E402
from air_alerts.baselines import BASELINES  # noqa: E402
from air_alerts.dataset import build_dataset  # noqa: E402
from air_alerts.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="evaluate on the held-out test period")
    args = parser.parse_args()

    df = build_dataset()
    if args.test:
        fit_on, target = df[df["split"].isin(["train", "val"])], df[df["split"] == "test"]
    else:
        fit_on, target = df[df["split"] == "train"], df[df["split"] == "val"]

    preds = {name: cls().fit(fit_on).predict(target) for name, cls in BASELINES.items()}
    results = evaluate(target, preds)

    period = "test" if args.test else "val"
    def span(part: pd.DataFrame) -> str:
        local = part.index.tz_convert(config.TZ)
        return f"{local.min():%Y-%m-%d %H:%M} .. {local.max():%Y-%m-%d %H:%M} Kyiv ({len(part)} rows)"

    print(f"Forecast issue times. Fit on {span(fit_on)}; scored on {period} {span(target)}; "
          f"horizon {config.HORIZON / pd.Timedelta(hours=1):.0f}h")
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        for name, part in results.groupby("slice", sort=False):
            print(f"\n[{name} hours]")
            print(part.drop(columns="slice").set_index("model").to_string())

    out = config.ROOT / "reports" / f"baselines_{period}.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nsaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
