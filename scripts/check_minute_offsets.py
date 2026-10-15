"""Do forecasts issued at arbitrary minutes score like the evaluated on-the-hour forecasts?

The model is trained on forecasts issued on the hour, but scripts/predict.py forecasts at any
minute. This check keeps everything else exactly as in the evaluation (final model, daily refit
at Kyiv midnight on hourly rows, test period) and moves each test forecast to a random minute
within its hour. Nothing is tuned on it: settings stay fixed, the result only says whether the
printed historical metrics carry over to off-the-hour use.

    python scripts/check_minute_offsets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_alerts import config  # noqa: E402
from air_alerts.data import build_episodes, load_events  # noqa: E402
from air_alerts.dataset import build_dataset  # noqa: E402
from air_alerts.evaluate import score  # noqa: E402
from air_alerts.features import make_features  # noqa: E402
from air_alerts.labels import label_new_start  # noqa: E402
from air_alerts.models import DECAY_MODELS  # noqa: E402
from air_alerts.predict import FINAL_MODEL  # noqa: E402
from air_alerts.walkforward import known_by  # noqa: E402


def main() -> None:
    events = load_events()
    episodes, _ = build_episodes(events)
    hourly = build_dataset(events)
    test_hours = hourly.index[hourly["split"] == "test"]

    rng = np.random.default_rng(0)
    shifted = test_hours + pd.to_timedelta(rng.integers(0, 60, len(test_hours)), unit="min")
    keep = shifted + config.HORIZON <= config.CUTOFF  # label window must lie inside the data
    variants = {"on the hour": test_hours[keep], "random minute": shifted[keep]}

    rows = []
    for name, times in variants.items():
        x = make_features(times, episodes)
        y = label_new_start(times, episodes["start"]).astype(int)
        p = np.empty(len(times))
        local_day = times.tz_convert(config.TZ).normalize()
        for day in local_day.unique():
            model = DECAY_MODELS[FINAL_MODEL]().fit(known_by(hourly, day))
            mask = np.asarray(local_day == day)
            p[mask] = model.predict(x[mask])
        rows.append({"forecast times": name, **score(y, p)})

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(f"Test period, model {FINAL_MODEL}, daily refit at Kyiv midnight")
        print(pd.DataFrame(rows).set_index("forecast times").to_string())


if __name__ == "__main__":
    main()
