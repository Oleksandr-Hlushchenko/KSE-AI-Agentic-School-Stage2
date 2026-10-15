"""Probabilistic scores for P(new alert start within the horizon)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """Brier and log loss judge calibrated probabilities; ROC/PR-AUC judge ranking only.

    `bss_vs_period_rate` compares with a constant equal to the evaluated period's own base rate.
    That constant is hindsight (unknowable at forecast time), so beating it means the forecast
    knows more than the period average. A baseline fitted on older data can be far worse than it
    when the alert rate shifts.
    """
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    rate = y.mean()
    brier = brier_score_loss(y, p)
    ref = rate * (1 - rate)
    both = 0 < rate < 1
    return {
        "n": len(y),
        "base_rate": rate,
        "mean_p": p.mean(),
        "brier": brier,
        "bss_vs_period_rate": 1 - brier / ref if ref > 0 else np.nan,
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "pr_auc": average_precision_score(y, p) if both else np.nan,
    }


def evaluate(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Scores on all hours and on calm hours (no alert active at t, so the model must
    anticipate an onset rather than ride an ongoing attack)."""
    slices = {"all": np.ones(len(df), bool), "calm": df["active_now"].to_numpy() == 0}
    rows = []
    for model, p in preds.items():
        for name, mask in slices.items():
            rows.append({"model": model, "slice": name, **score(df["y"].to_numpy()[mask], p[mask])})
    return pd.DataFrame(rows)
