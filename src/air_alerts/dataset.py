"""One row per forecast issue time: features, target `y` and split."""
import pandas as pd

from . import config
from .data import build_episodes, load_events
from .features import make_features
from .labels import assign_split, forecast_times, label_new_start


def build_dataset(events: pd.DataFrame | None = None, cutoff: pd.Timestamp = config.CUTOFF) -> pd.DataFrame:
    """Rows whose label window ends by `cutoff`, i.e. whose label is known at `cutoff`."""
    if events is None:
        events = load_events()
    # Episodes come from the whole log (2022 included) so that features on 2023-01-01 have history.
    episodes, _ = build_episodes(events)
    times = forecast_times(config.DATA_START, cutoff)
    df = make_features(times, episodes)
    df["y"] = label_new_start(times, episodes["start"]).astype(int)
    df["split"] = assign_split(times)
    df.index.name = "t"
    return df
