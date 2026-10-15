"""Learned models with the same fit/predict interface as the baselines.

Hyperparameters are fixed up front, not tuned on validation: there are only 574 validation
hours, and tuning on them would turn validation into training data.
"""
from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import RecentRatePoisson24h
from .features import FEATURE_COLUMNS

LOG_COLUMNS = [
    "starts_3h", "starts_6h", "starts_24h", "starts_7d",
    "active_min_24h", "active_min_7d", "current_alert_min",
    "min_since_last_start", "min_since_last_end",
]
# "Never happened" (NaN) is treated as a very long time ago: ~70 days.
NEVER_MIN = 1e5


def baseline_logit(df: pd.DataFrame) -> np.ndarray:
    """Log-odds of the 24h recent-rate baseline, as a feature. With it, a model only has to
    learn corrections to the baseline instead of rebuilding the intensity signal itself.
    Clipped because 0 starts in 24h gives P = 0 (log-odds -inf)."""
    p = np.clip(RecentRatePoisson24h().predict(df), 0.01, 0.99)
    return np.log(p / (1 - p))


def decay_weights(index: pd.DatetimeIndex, half_life_days: float | None) -> np.ndarray | None:
    """Sample weights halving every `half_life_days`, counted back from the newest training row.

    Only the training rows' own timestamps are used, so the weights carry no outside information.
    """
    if half_life_days is None:
        return None
    age_days = (index.max() - index) / pd.Timedelta(days=1)
    return np.power(0.5, np.asarray(age_days, dtype=float) / half_life_days)


def logistic_design(df: pd.DataFrame, use_baseline: bool = False) -> pd.DataFrame:
    """log1p of counts and durations, so that the log-odds respond to *relative* change in
    alert intensity, and that response keeps going beyond the range seen in training."""
    x = pd.DataFrame(index=df.index)
    for c in LOG_COLUMNS:
        x[f"log_{c}"] = np.log1p(df[c].fillna(NEVER_MIN))
    x["active_now"] = df["active_now"]
    x["hour_sin"] = df["hour_sin"]
    x["hour_cos"] = df["hour_cos"]
    if use_baseline:
        x["baseline_logit"] = baseline_logit(df)
    return x


def tree_design(df: pd.DataFrame, use_baseline: bool = False) -> pd.DataFrame:
    x = df[FEATURE_COLUMNS]
    return x.assign(baseline_logit=baseline_logit(df)) if use_baseline else x


class LogisticRate:
    def __init__(self, use_baseline: bool = False, half_life_days: float | None = None):
        self.use_baseline = use_baseline
        self.half_life_days = half_life_days

    def fit(self, train: pd.DataFrame):
        # The scaler is part of the pipeline, so its mean/std come from training rows only.
        self.pipe_ = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        w = decay_weights(train.index, self.half_life_days)
        self.pipe_.fit(logistic_design(train, self.use_baseline), train["y"], logisticregression__sample_weight=w)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(logistic_design(df, self.use_baseline))[:, 1]

    def coefficients(self) -> pd.Series:
        lr = self.pipe_[-1]
        names = self.pipe_[0].get_feature_names_out()
        return pd.Series(lr.coef_[0], index=names).sort_values(key=abs, ascending=False)


class LightGBM:
    """Gradient-boosted trees on the raw features. Same settings as BoostedTrees below.

    Needs the OpenMP runtime on macOS (`brew install libomp`).
    """

    def __init__(self, use_baseline: bool = False, half_life_days: float | None = None):
        self.use_baseline = use_baseline
        self.half_life_days = half_life_days

    def fit(self, train: pd.DataFrame):
        self.model_ = lgb.LGBMClassifier(
            learning_rate=0.05,
            n_estimators=300,
            num_leaves=15,
            min_child_samples=50,
            reg_lambda=1.0,
            random_state=0,
            n_jobs=1,  # deterministic results
            verbose=-1,
        )
        w = decay_weights(train.index, self.half_life_days)
        self.model_.fit(tree_design(train, self.use_baseline), train["y"], sample_weight=w)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(tree_design(df, self.use_baseline))[:, 1]


class BoostedTrees:
    """scikit-learn's histogram gradient boosting (inspired by LightGBM): a cross-check that
    the LightGBM result is not an implementation quirk.

    Built-in early stopping is off: it holds out a *random* subset of training rows, which
    mixes time order, and makes results depend on the seed.
    """

    def fit(self, train: pd.DataFrame):
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
        self.model_.fit(train[FEATURE_COLUMNS], train["y"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(df[FEATURE_COLUMNS])[:, 1]


MODELS = {
    "logistic": LogisticRate,
    "lightgbm": LightGBM,
    "sklearn_gbdt": BoostedTrees,
    "logistic+base": partial(LogisticRate, use_baseline=True),
    "lightgbm+base": partial(LightGBM, use_baseline=True),
}

# Time-decay variants, meant for walk-forward: half-lives fixed in advance, not tuned.
DECAY_MODELS = {
    f"{name}+decay{h}": partial(cls, half_life_days=h)
    for h in (14, 60)
    for name, cls in (("logistic", LogisticRate), ("lightgbm", LightGBM))
}
