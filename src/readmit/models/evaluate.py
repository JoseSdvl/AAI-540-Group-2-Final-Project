"""Evaluation metrics + subgroup breakdowns.

Kept dependency-light (numpy + sklearn only) so it can run inside the
SageMaker training and processing containers without extra installs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the headline metric dict written to ``evaluation.json``."""
    y_pred = (y_proba >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(tp + fn),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k_fraction: float) -> float:
    """Recall when we flag the top ``k_fraction`` of the population as high risk.

    This is the metric care-management teams actually plan against — they have
    capacity for only so many follow-up calls, so what matters is "of the
    patients we *will* call, how many true readmits are we catching?"
    """
    if not 0 < k_fraction <= 1:
        raise ValueError("k_fraction must be in (0, 1]")
    n = len(y_proba)
    k = max(1, int(np.ceil(n * k_fraction)))
    order = np.argsort(-y_proba)
    flagged = np.zeros(n, dtype=bool); flagged[order[:k]] = True
    pos = int(y_true.sum())
    if pos == 0:
        return 0.0
    return float(((flagged) & (y_true == 1)).sum() / pos)


def subgroup_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    group_cols: Iterable[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-subgroup metric table for fairness/bias review."""
    rows: list[dict] = []
    for col in group_cols:
        if col not in df.columns:
            continue
        for value, idx in df.groupby(col).indices.items():
            if len(idx) < 30 or np.unique(y_true[idx]).size < 2:
                # Skip small/degenerate cells where metrics are unstable.
                continue
            m = compute_metrics(y_true[idx], y_proba[idx], threshold=threshold)
            rows.append({"group": col, "value": value, **m})
    return pd.DataFrame(rows)
