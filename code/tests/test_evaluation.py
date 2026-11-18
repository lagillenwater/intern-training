import numpy as np
import pytest
from sklearn.metrics import f1_score

from helpers import SPECIES_ORDER
from helpers.evaluation import (
    bootstrap_ci,
    macro_f1_matrix,
    paired_bootstrap_diff,
    paired_differences,
    per_class_f1,
    score_table,
)


def test_macro_f1_matrix_matches_sklearn():
    rng = np.random.default_rng(0)
    yt, yp = rng.integers(1, 8, (50, 200)), rng.integers(1, 8, (50, 200))
    labels = list(range(1, 8))
    expected = [
        f1_score(yt[i], yp[i], labels=labels, average="macro", zero_division=0) for i in range(50)
    ]
    np.testing.assert_allclose(macro_f1_matrix(yt, yp, labels), expected)


def test_bootstrap_ci_perfect():
    y = np.array([1, 2, 3, 2, 1, 3] * 20)
    assert bootstrap_ci(y, y) == (1.0, 1.0)


def test_bootstrap_ci_brackets_and_deterministic():
    rng = np.random.default_rng(1)
    yt = rng.integers(1, 8, 500)
    yp = np.where(rng.random(500) < 0.7, yt, rng.integers(1, 8, 500))
    lo, hi = bootstrap_ci(yt, yp)
    assert lo < f1_score(yt, yp, average="macro") < hi
    assert bootstrap_ci(yt, yp) == (lo, hi)


def test_paired_diff_identical_is_zero():
    yt = np.array([1, 2, 3] * 50)
    assert paired_bootstrap_diff(yt, yt, yt) == {"diff": 0.0, "low": 0.0, "high": 0.0}


def test_paired_diff_detects_better_model():
    rng = np.random.default_rng(2)
    yt = rng.integers(1, 8, 800)
    weak = np.where(rng.random(800) < 0.4, yt, rng.integers(1, 8, 800))
    strong = np.where(rng.random(800) < 0.9, yt, rng.integers(1, 8, 800))
    r = paired_bootstrap_diff(yt, weak, strong)
    assert r["diff"] > 0 and r["low"] > 0


def test_score_table(preds_fixture):
    t = score_table(preds_fixture)
    assert list(t.columns) == ["species", "model", "macro_f1", "accuracy", "ci_low", "ci_high"]
    assert len(t) == 15 and list(dict.fromkeys(t.species)) == SPECIES_ORDER
    y = preds_fixture[preds_fixture.species == "yeast"]
    got = t.query("species == 'yeast' and model == 'baseline'").macro_f1.item()
    assert got == pytest.approx(f1_score(y.y_true, y.y_pred_baseline, average="macro", zero_division=0))
    assert (t.ci_low <= t.macro_f1).all() and (t.macro_f1 <= t.ci_high).all()


def test_paired_differences_columns(preds_fixture):
    d = paired_differences(preds_fixture)
    assert list(d.columns) == ["species", "diff", "low", "high", "excludes_zero"]
    assert list(d.species) == SPECIES_ORDER
    assert d.excludes_zero.all()


def test_per_class_f1(preds_fixture):
    c = per_class_f1(preds_fixture)
    assert list(c.columns) == ["species", "model", "ec_class", "class_name", "f1"]
    assert 7 not in set(c.loc[c.species == "yeast", "ec_class"])
    h = preds_fixture[preds_fixture.species == "human"]
    got = c.query("species == 'human' and model == 'finetuned' and ec_class == 2").f1.item()
    assert got == pytest.approx(
        f1_score(h.y_true, h.y_pred_finetuned, labels=[2], average="macro", zero_division=0)
    )
    assert c.query("ec_class == 2").class_name.iloc[0] == "Transferases"
