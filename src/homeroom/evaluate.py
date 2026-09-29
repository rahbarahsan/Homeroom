"""Metrics: accuracy, macro-F1, expected calibration error (ECE), NLL, confusion pairs."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import f1_score


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(1)
    correct = (proba.argmax(1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def compute_metrics(y_true, proba) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(1)
    p_true = np.clip(proba[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return {
        "n": int(len(y_true)),
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=np.unique(y_true), zero_division=0)),
        "ece": ece(y_true, proba),
        "nll": float(-np.log(p_true).mean()),
    }


def top_confusions(y_true, y_pred, k: int) -> list[tuple[int, int, int]]:
    """Most frequent (true, predicted) error pairs: [(true, pred, count), ...]."""
    c = Counter((int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p)
    return [(t, p, n) for (t, p), n in c.most_common(k)]
