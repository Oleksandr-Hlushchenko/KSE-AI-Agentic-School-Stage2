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


class FlooredRecentRate24h(RecentRatePoisson24h):
    """The 24h rate with probabilities kept in [0.05, 0.95].

    With no start in the last 24h the Poisson formula says P = 0, which is never justified
    (on validation 4 of 16 such hours were followed by an alert). The floor fixes that
    structural flaw. It was chosen after seeing validation, as a round number, not tuned.
    """

    floor = 0.05

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(super().predict(df), self.floor, 1 - self.floor)


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
    "recent_rate_24h_floor": FlooredRecentRate24h,
}
