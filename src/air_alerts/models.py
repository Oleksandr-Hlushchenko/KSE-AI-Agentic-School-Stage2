"""Learned models with the same fit/predict interface as the baselines.

Hyperparameters are fixed up front, not tuned on validation: there are only 574 validation
hours, and tuning on them would turn validation into training data.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

LOG_COLUMNS = [
    "starts_3h", "starts_6h", "starts_24h", "starts_7d",
    "active_min_24h", "active_min_7d", "current_alert_min",
    "min_since_last_start", "min_since_last_end",
]
# "Never happened" (NaN) is treated as a very long time ago: ~70 days.
NEVER_MIN = 1e5


def logistic_design(df: pd.DataFrame) -> pd.DataFrame:
    """log1p of counts and durations, so that the log-odds respond to *relative* change in
    alert intensity, and that response keeps going beyond the range seen in training."""
    x = pd.DataFrame(index=df.index)
    for c in LOG_COLUMNS:
        x[f"log_{c}"] = np.log1p(df[c].fillna(NEVER_MIN))
    x["active_now"] = df["active_now"]
    x["hour_sin"] = df["hour_sin"]
    x["hour_cos"] = df["hour_cos"]
    return x


class LogisticRate:
    def fit(self, train: pd.DataFrame):
        # The scaler is part of the pipeline, so its mean/std come from training rows only.
        self.pipe_ = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        self.pipe_.fit(logistic_design(train), train["y"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(logistic_design(df))[:, 1]

    def coefficients(self) -> pd.Series:
        lr = self.pipe_[-1]
        names = self.pipe_[0].get_feature_names_out()
        return pd.Series(lr.coef_[0], index=names).sort_values(key=abs, ascending=False)


class LightGBM:
    """Gradient-boosted trees on the raw features. Same settings as BoostedTrees below.

    Needs the OpenMP runtime on macOS (`brew install libomp`).
    """

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
        self.model_.fit(train[FEATURE_COLUMNS], train["y"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(df[FEATURE_COLUMNS])[:, 1]


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
}
