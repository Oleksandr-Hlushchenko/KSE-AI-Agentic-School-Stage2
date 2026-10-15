"""Reference forecasters. Any real model has to beat these to be worth anything.

Each one has `fit(train_df)` and `predict(df)`, and returns P(new alert start within the horizon).
`fit` may only look at the rows it is given; the caller passes training rows only.
"""
import numpy as np
import pandas as pd

from . import config

HORIZON_H = config.HORIZON / pd.Timedelta(hours=1)


class TrainBaseRate:
    """The same probability for every hour: the training-period frequency."""

    def fit(self, train: pd.DataFrame):
        self.p_ = train["y"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.p_)


class HourOfDayClimatology:
    """Training-period frequency for each Kyiv hour of day."""

    def fit(self, train: pd.DataFrame):
        self.p_ = train.groupby("hour_kyiv")["y"].mean()
        self.default_ = train["y"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["hour_kyiv"].map(self.p_).fillna(self.default_).to_numpy()


class RecentRatePoisson:
    """No fitting: treat starts as a Poisson process whose rate is the last `window` count.

    P(at least one start in H hours) = 1 - exp(-rate * H). Adapts to escalation immediately,
    which the training-period baselines cannot.
    """

    window = "7d"
    window_h = 168

    def fit(self, train: pd.DataFrame):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        rate = df[f"starts_{self.window}"].to_numpy() / self.window_h
        return 1 - np.exp(-rate * HORIZON_H)


class RecentRatePoisson24h(RecentRatePoisson):
    window = "24h"
    window_h = 24


class ConditionalPersistence:
    """P(start in next 3h | was there a start in the last 3h), estimated on training rows."""

    def fit(self, train: pd.DataFrame):
        recent = train["starts_3h"] > 0
        self.p_ = train.groupby(recent)["y"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["starts_3h"] > 0).map(self.p_).to_numpy()


BASELINES = {
    "train_base_rate": TrainBaseRate,
    "hour_climatology": HourOfDayClimatology,
    "persistence_3h": ConditionalPersistence,
    "recent_rate_7d": RecentRatePoisson,
    "recent_rate_24h": RecentRatePoisson24h,
}
