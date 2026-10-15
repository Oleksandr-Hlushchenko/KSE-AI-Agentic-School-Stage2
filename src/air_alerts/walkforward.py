"""Walk-forward evaluation: refit once per Kyiv day, as a deployed forecaster would.

The label of a row issued at t is known only at t + horizon. So before day D starts, the model
may train on rows with t + horizon <= D, and it then forecasts every hour of D. Earlier days of
the evaluated period become training data for later ones; nothing from D or after is used.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import config


def known_by(df: pd.DataFrame, moment: pd.Timestamp, horizon: pd.Timedelta = config.HORIZON) -> pd.DataFrame:
    """Rows whose label has been observed by `moment`."""
    return df[df.index + horizon <= moment]


def walk_forward_predict(
    df: pd.DataFrame,
    factory: Callable,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: pd.Timedelta = config.HORIZON,
) -> pd.Series:
    """Forecasts for rows with start <= t < end, refitting at every Kyiv midnight."""
    preds = pd.Series(np.nan, index=df.index)
    day = start.tz_convert(config.TZ).normalize()
    while day < end:
        next_day = day + pd.DateOffset(days=1)  # calendar day, so DST days stay aligned
        target = df[(df.index >= max(day, start)) & (df.index < min(next_day, end))]
        if len(target):
            model = factory().fit(known_by(df, day, horizon))
            preds.loc[target.index] = model.predict(target)
        day = next_day
    return preds[(df.index >= start) & (df.index < end)]
