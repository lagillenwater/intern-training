import matplotlib.pyplot as plt
import numpy as np

from helpers.data import SPECIES_ORDER
from helpers.plots import (
    compute_umap,
    plot_confusion,
    plot_confusion_grid,
    plot_distance_curve,
    plot_umap,
)


def test_distance_curve(results_fixture, divergence_fixture):
    fig = plot_distance_curve(results_fixture, divergence_fixture)
    ax = fig.axes[0]
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert [t.split("\n")[0] for t in ticks] == SPECIES_ORDER
    assert ticks[0] == "human\n(held-out)"
    assert ticks[1].startswith("mouse\n~") and ticks[1].endswith(" Mya")
    assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] >= 1
    assert {"baseline", "finetuned"} <= {t.get_text() for t in ax.get_legend().get_texts()}
    plt.close(fig)


def test_confusion_axes():
    fig = plot_confusion([1, 2, 3, 7], [1, 2, 2, 7], "x")
    assert "Hydrolases" in [t.get_text() for t in fig.axes[0].get_yticklabels()]
    plt.close(fig)


def test_confusion_grid(preds_fixture):
    fig = plot_confusion_grid(preds_fixture)
    assert len([a for a in fig.axes if a.get_title()]) == 10
    plt.close(fig)


def test_umap_deterministic():
    X = np.random.default_rng(0).normal(size=(200, 16))
    a, b = compute_umap(X, seed=0), compute_umap(X, seed=0)
    assert a.shape == (200, 2) and np.allclose(a, b)


def _legend_colors(fig):
    leg = fig.axes[0].get_legend()
    return {t.get_text(): h.get_facecolor()[0].tolist() for t, h in zip(leg.get_texts(), leg.legend_handles)}


def test_umap_colors_stable():
    c = np.zeros((4, 2))
    f1 = plot_umap(c, ["yeast", "human", "yeast", "fly"], "a")
    f2 = plot_umap(c, ["fly", "yeast", "yeast", "yeast"], "b")
    assert _legend_colors(f1)["yeast"] == _legend_colors(f2)["yeast"]
    plt.close("all")


def test_umap_class_labels():
    fig = plot_umap(np.zeros((3, 2)), [1, 3, 3], "c")
    assert set(_legend_colors(fig)) == {"Oxidoreductases", "Hydrolases"}
    plt.close(fig)


def test_distance_curve_neutral_title(results_fixture, divergence_fixture):
    fig = plot_distance_curve(results_fixture, divergence_fixture)
    assert "drop" not in fig.axes[0].get_title(loc="left").lower()
    fig2 = plot_distance_curve(results_fixture, divergence_fixture, title="Custom")
    assert fig2.axes[0].get_title(loc="left") == "Custom"
    plt.close("all")


def test_umap_class_markers_distinct():
    from helpers.plots import CLASS_COLORS, CLASS_MARKERS

    assert len(set(CLASS_MARKERS.values())) == 7 and len(set(CLASS_COLORS.values())) == 7
