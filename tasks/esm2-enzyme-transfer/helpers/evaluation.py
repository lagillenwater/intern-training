"""Scoring helpers: macro-F1, bootstrap intervals, and paired comparisons.

All bootstrap work is vectorized: one (n_boot, n) matrix of resampled row
indices is drawn, and every resample is scored at once.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from helpers.data import CLASS_NAMES, SPECIES_ORDER


def _class_f1_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Per-class F1 for inputs of shape (B, n); output shape (B, K).

    F1 is 0 where a class has no true and no predicted members."""
    classes = np.asarray(labels)
    true_hit = np.asarray(y_true)[..., None] == classes  # (B, n, K)
    pred_hit = np.asarray(y_pred)[..., None] == classes
    tp = (true_hit & pred_hit).sum(axis=1)
    fp = (~true_hit & pred_hit).sum(axis=1)
    fn = (true_hit & ~pred_hit).sum(axis=1)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros(denom.shape), where=denom > 0)


def macro_f1_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Macro-F1 for each row of (B, n) label arrays; returns shape (B,)."""
    return _class_f1_matrix(y_true, y_pred, labels).mean(axis=1)


def _labels(*arrays: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([np.asarray(a).ravel() for a in arrays]))


def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return float(low), float(high)


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap interval for macro-F1, resampling proteins."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = _labels(y_true, y_pred)
    idx = _boot_indices(len(y_true), n_boot, seed)
    return _interval(macro_f1_matrix(y_true[idx], y_pred[idx], labels), level)


def paired_bootstrap_diff(
    y_true, y_pred_a, y_pred_b, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> dict:
    """Macro-F1 of model b minus model a, with a paired percentile interval.

    Both models are scored on the same resampled proteins in every draw."""
    y_true, y_pred_a, y_pred_b = np.asarray(y_true), np.asarray(y_pred_a), np.asarray(y_pred_b)
    labels = _labels(y_true, y_pred_a, y_pred_b)
    point = macro_f1_matrix(y_true[None], y_pred_b[None], labels)[0] - macro_f1_matrix(
        y_true[None], y_pred_a[None], labels
    )[0]
    idx = _boot_indices(len(y_true), n_boot, seed)
    diffs = macro_f1_matrix(y_true[idx], y_pred_b[idx], labels) - macro_f1_matrix(
        y_true[idx], y_pred_a[idx], labels
    )
    low, high = _interval(diffs, level)
    return {"diff": float(point), "low": low, "high": high}


def score_table(
    preds: pd.DataFrame,
    models: Sequence[str] = ("floor", "baseline", "finetuned"),
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Macro-F1, accuracy, and bootstrap interval per species and model."""
    rows = []
    for species, group in preds.groupby("species", sort=False):
        y_true = group["y_true"].to_numpy()
        for model in models:
            y_pred = group[f"y_pred_{model}"].to_numpy()
            low, high = bootstrap_ci(y_true, y_pred, n_boot, seed)
            f1 = macro_f1_matrix(y_true[None], y_pred[None], _labels(y_true, y_pred))[0]
            rows.append((species, model, float(f1), float((y_true == y_pred).mean()), low, high))
    table = pd.DataFrame(rows, columns=["species", "model", "macro_f1", "accuracy", "ci_low", "ci_high"])
    return _order_species(table)


def paired_differences(
    preds: pd.DataFrame, a: str = "baseline", b: str = "finetuned", n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Paired bootstrap of model b minus model a for each species."""
    rows = []
    for species, group in preds.groupby("species", sort=False):
        r = paired_bootstrap_diff(
            group["y_true"], group[f"y_pred_{a}"], group[f"y_pred_{b}"], n_boot, seed
        )
        rows.append((species, r["diff"], r["low"], r["high"], bool(r["low"] > 0 or r["high"] < 0)))
    table = pd.DataFrame(rows, columns=["species", "diff", "low", "high", "excludes_zero"])
    return _order_species(table)


def per_class_f1(preds: pd.DataFrame, models: Sequence[str] = ("baseline", "finetuned")) -> pd.DataFrame:
    """F1 for each EC class, species, and model; classes absent from truth are omitted."""
    frames = []
    for species, group in preds.groupby("species", sort=False):
        y_true = group["y_true"].to_numpy()
        present = np.unique(y_true)
        for model in models:
            y_pred = group[f"y_pred_{model}"].to_numpy()
            f1 = _class_f1_matrix(y_true[None], y_pred[None], present)[0]
            frames.append(
                pd.DataFrame({"species": species, "model": model, "ec_class": present, "f1": f1})
            )
    table = pd.concat(frames, ignore_index=True)
    table["class_name"] = table["ec_class"].map(CLASS_NAMES)
    return _order_species(table[["species", "model", "ec_class", "class_name", "f1"]])


def _order_species(table: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by SPECIES_ORDER, keeping the existing order within a species."""
    rank = table["species"].map({s: i for i, s in enumerate(SPECIES_ORDER)})
    return table.iloc[np.argsort(rank.to_numpy(), kind="stable")].reset_index(drop=True)
