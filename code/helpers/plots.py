"""Figures for the final presentation: distance curve, confusion matrices, and UMAPs.

Colors follow the dataviz reference palette (light mode). Species and classes use
fixed mappings, so a category has the same color in every figure.
"""
import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from helpers.data import CLASS_NAMES, SPECIES_ORDER

# Categorical slots 1-5 and 1-7 of the reference palette, in fixed order.
SPECIES_COLORS = dict(zip(SPECIES_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
CLASS_COLORS = dict(zip(CLASS_NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#5b3a29", "#6250d6"]))
# Class marker shapes, so classes with similar colors stay distinguishable in a dense cloud.
CLASS_MARKERS = dict(zip(CLASS_NAMES, ["o", "s", "^", "D", "v", "P", "h"]))
# Models are a different kind of category from species, so they avoid the species hues.
MODEL_COLORS = {"floor": "#8a8984", "baseline": "#52514e", "finetuned": "#6250d6"}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"


def _style_axes(ax) -> None:
    """Quiet frame: no top/right spines, light grid."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT)


def plot_distance_curve(
    results: pd.DataFrame,
    divergence: pd.DataFrame,
    models: Sequence[str] = ("baseline", "finetuned"),
    title: str = "Macro-F1 by species, ordered by distance from human",
) -> Figure:
    """Plot macro-F1 (with 95% intervals) against species, ordered by evolutionary distance."""
    mya = divergence.set_index("species")["mya"].reindex(SPECIES_ORDER)
    ticks = [
        "human\n(held-out)" if s == "human" else f"{s}\n~{int(mya[s]):,} Mya" for s in SPECIES_ORDER
    ]
    x = np.arange(len(SPECIES_ORDER))
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.linspace(-0.04, 0.04, len(models))
    for model, dx in zip(models, offsets):
        rows = results[results["model"] == model].set_index("species").reindex(SPECIES_ORDER)
        y = rows["macro_f1"].to_numpy()
        err = np.vstack([y - rows["ci_low"].to_numpy(), rows["ci_high"].to_numpy() - y])
        ax.errorbar(
            x + dx, y, yerr=err, label=model,
            color=MODEL_COLORS.get(model, INK_SOFT), linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, capsize=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Species, from closest to furthest from human", color=INK_SOFT)
    ax.set_ylabel("Macro-F1 (higher is better)", color=INK_SOFT)
    ax.set_title(title, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(title="Model (bars show 95% interval)", frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title: str, ax=None, annotate: bool = True) -> Figure:
    """Plot a row-normalized confusion matrix: each row shows where one true class was sent."""
    labels = list(CLASS_NAMES)
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals > 0)
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    names = list(CLASS_NAMES.values())
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted class", color=INK_SOFT)
    ax.set_ylabel("True class", color=INK_SOFT)
    ax.set_title(title, color=INK, fontsize=10)
    if annotate:
        for (i, j), v in np.ndenumerate(share):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else INK)
    if own_figure:
        ax.figure.tight_layout()
    return ax.figure


def plot_confusion_grid(preds: pd.DataFrame, models: Sequence[str] = ("baseline", "finetuned")) -> Figure:
    """Plot one confusion matrix per species (rows) and model (columns).

    The floor model is left out because it predicts a single class.
    """
    fig, axes = plt.subplots(
        len(SPECIES_ORDER), len(models), figsize=(4.6 * len(models), 4.4 * len(SPECIES_ORDER)), squeeze=False
    )
    by_species = {s: g for s, g in preds.groupby("species")}
    cells = [(r, c, s, m) for r, s in enumerate(SPECIES_ORDER) for c, m in enumerate(models)]
    for r, c, species, model in cells:
        rows = by_species[species]
        plot_confusion(rows["y_true"], rows[f"y_pred_{model}"], f"{species} - {model}", ax=axes[r, c], annotate=False)
    fig.tight_layout()
    return fig


def compute_umap(embeddings: np.ndarray, seed: int = 0, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Squash embeddings to 2 dimensions with UMAP. The same seed gives the same picture."""
    import umap

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed, n_jobs=1)
    with warnings.catch_warnings():
        # Newer umap-learn warns about a future default layout; the current one is deterministic.
        warnings.filterwarnings("ignore", message="compatibility_layout", category=FutureWarning)
        return reducer.fit_transform(np.asarray(embeddings))


def plot_umap(coords: np.ndarray, labels, title: str, ax=None) -> Figure:
    """Scatter 2-D coordinates colored by species or by enzyme class.

    Labels that are species names use the species colors; integer labels 1-7 use the
    class colors. Each category always gets the same color (and, for classes, the same marker shape).
    """
    labels = pd.Series(np.asarray(labels, dtype=object))
    if labels.isin(SPECIES_COLORS).all():
        colors, names = SPECIES_COLORS, {s: s for s in SPECIES_ORDER}
        markers = dict.fromkeys(SPECIES_ORDER, "o")
    else:
        labels = labels.astype(int)
        colors, names, markers = CLASS_COLORS, CLASS_NAMES, CLASS_MARKERS
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    coords = np.asarray(coords)
    present = [k for k in colors if (labels == k).any()]
    for key in present:
        mask = (labels == key).to_numpy()
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=18, color=colors[key], marker=markers[key], alpha=0.85,
            edgecolors="white", linewidths=0.3, label=names[key],
        )
    ax.set_xlabel("UMAP 1 (axes have no units)", color=INK_SOFT)
    ax.set_ylabel("UMAP 2", color=INK_SOFT)
    ax.set_title(title, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axes(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=1.8)
    ax.figure.tight_layout()
    return ax.figure
