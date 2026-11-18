"""Tests for the notebook check functions.

One table (the `cases` fixture) drives the tests: for every check it holds a
correct call and a deliberately wrong call. Other tests cover unfilled input,
messages, and robustness (no traceback ever escapes).
"""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import checks
from checks import *  # noqa: F403  (the notebooks do the same)
from helpers import MAX_RESIDUES, SPECIES_ORDER, ec_classes, load_dataset, score_table

CHECK_NAMES = [
    "check_species_list",
    "check_ec_class_function",
    "check_species_counts",
    "check_mean_length",
    "check_class_counts",
    "check_majority_accuracy",
    "check_fraction_truncated",
    "check_multiclass_count",
    "check_train_arrays",
    "check_fitted_classifier",
    "check_macro_f1",
    "check_prediction_table",
    "check_hyperparameters",
    "check_saved_model",
    "check_species_f1",
    "check_results_table",
    "check_umap_coords",
    "check_misclassified",
]


@pytest.fixture(scope="module")
def df():
    return load_dataset()


def _real_ec_class(ec):
    return min(ec_classes(ec))


def _saved_model(tmp_path):
    for name in ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"]:
        (tmp_path / name).write_text("x")
    return tmp_path


@pytest.fixture
def cases(df, preds_fixture, results_fixture, raw_fixture, tmp_path):
    """name -> (good_args, bad_args)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(df), 4))
    train = (df["split"] == "train").to_numpy()
    y = df["ec_class"].to_numpy()
    human_train = df[(df["species"] == "human") & (df["split"] == "train")]
    test = df[df["split"] == "test"]
    test_preds = pd.DataFrame(
        {
            "accession": test["accession"],
            "species": test["species"],
            "y_true": test["ec_class"],
            "y_pred_baseline": 1,
        }
    )
    clf = LogisticRegression().fit(X[train], y[train])
    yeast = preds_fixture[preds_fixture["species"] == "yeast"]
    yeast_err = preds_fixture[
        preds_fixture["species"].isin(["yeast", "ecoli"])
        & (preds_fixture["y_true"] != preds_fixture["y_pred_finetuned"])
    ]
    misclassified = yeast_err[["accession", "species", "y_true", "y_pred_finetuned"]].assign(
        uniprot_url="https://www.uniprot.org/uniprotkb/" + yeast_err["accession"]
    )
    bad_missing = tmp_path / "empty"
    bad_missing.mkdir()
    good_model = tmp_path / "model"
    good_model.mkdir()
    _saved_model(good_model)
    good_f1 = float(results_fixture.query("species == 'yeast' and model == 'baseline'")["macro_f1"].iloc[0])
    return {
        "check_species_list": ((list(SPECIES_ORDER[::-1]),), (["human", "mouse"],)),
        "check_ec_class_function": ((_real_ec_class,), (lambda ec: 1,)),
        "check_species_counts": ((df["species"].value_counts().sort_index(), df), (df["species"].value_counts() + 1, df)),
        "check_mean_length": (
            (float(df.loc[df["species"] == "yeast", "length"].mean()), df),
            (float(df["length"].mean()), df),
        ),
        "check_class_counts": ((pd.crosstab(df["species"], df["ec_class"]).iloc[::-1], df), (pd.crosstab(df["species"], df["ec_class"]) + 1, df)),
        "check_majority_accuracy": (
            (float(human_train["ec_class"].value_counts(normalize=True).max()), df),
            (0.5, df),
        ),
        "check_fraction_truncated": ((float((df["length"] > MAX_RESIDUES).mean()), df), (0.9, df)),
        "check_multiclass_count": ((1, raw_fixture["human"]), (5, raw_fixture["human"])),
        "check_train_arrays": ((X[train], y[train], X, df), (X, y, X, df)),
        "check_fitted_classifier": ((clf,), (LogisticRegression(),)),
        "check_macro_f1": ((1.0, [1, 2, 3], [1, 2, 3]), (0.2, [1, 2, 3], [1, 2, 3])),
        "check_prediction_table": ((test_preds.sample(frac=1, random_state=0), df, "baseline"), (test_preds.iloc[:10], df, "baseline")),
        "check_hyperparameters": ((3, 5e-5), (100, 5e-5)),
        "check_saved_model": ((good_model,), (bad_missing,)),
        "check_species_f1": ((good_f1, preds_fixture, "yeast", "baseline"), (good_f1 + 0.1, preds_fixture, "yeast", "baseline")),
        "check_results_table": ((results_fixture.iloc[::-1], preds_fixture), (results_fixture.assign(macro_f1=0.5), preds_fixture)),
        "check_umap_coords": ((rng.normal(size=(10, 2)), 10), (rng.normal(size=(10, 3)), 10)),
        "check_misclassified": ((misclassified.iloc[::-1], preds_fixture), (misclassified.iloc[:3], preds_fixture)),
    }


def test_every_check_is_exported():
    assert sorted(checks.__all__) == sorted(CHECK_NAMES)


@pytest.mark.parametrize("name", CHECK_NAMES)
def test_correct_input_passes(name, cases, capsys):
    good, _ = cases[name]
    assert getattr(checks, name)(*good) is True
    out = capsys.readouterr().out
    assert out.startswith("Check passed:")
    assert out.count("\n") == 1


@pytest.mark.parametrize("name", CHECK_NAMES)
def test_wrong_input_fails(name, cases, capsys):
    _, bad = cases[name]
    assert getattr(checks, name)(*bad) is False
    out = capsys.readouterr().out
    assert out.startswith("Not yet:")
    assert out.count("\n") == 1


@pytest.mark.parametrize("unfilled", [None, ...], ids=["None", "Ellipsis"])
@pytest.mark.parametrize("name", CHECK_NAMES)
def test_checks_handle_unfilled(name, unfilled, cases, capsys):
    good, _ = cases[name]
    assert getattr(checks, name)(unfilled, *good[1:]) is False
    assert capsys.readouterr().out.startswith("Not yet: fill in")


def test_unfilled_message_uses_variable_name(capsys):
    checks.check_train_arrays(None, None, None, None)
    assert capsys.readouterr().out.startswith("Not yet: fill in X_train first.")
    checks.check_fitted_classifier(...)
    assert capsys.readouterr().out.startswith("Not yet: fill in clf first.")
    checks.check_species_counts(None, None)
    assert capsys.readouterr().out.startswith("Not yet: fill in species_counts first.")


@pytest.mark.parametrize("name", CHECK_NAMES)
@pytest.mark.parametrize("junk", ["text", 3.5, [1, 2], {"a": 1}, object()], ids=["str", "float", "list", "dict", "object"])
def test_no_traceback_escapes(name, junk, cases, capsys):
    good, _ = cases[name]
    result = getattr(checks, name)(junk, *good[1:])
    out = capsys.readouterr().out
    assert result in (True, False)
    assert out.startswith(("Check passed:", "Not yet:")) and out.count("\n") == 1


def test_function_that_raises_is_reported(capsys):
    def broken(ec):
        raise KeyError("boom")

    assert check_ec_class_function(broken) is False
    out = capsys.readouterr().out
    assert out.startswith("Not yet: your function raised KeyError:")
    assert 'for input "' in out


def test_ec_function_must_return_int(capsys):
    assert check_ec_class_function(lambda ec: str(_real_ec_class(ec))) is False
    assert "int" in capsys.readouterr().out


def test_ec_function_not_callable(capsys):
    assert check_ec_class_function(3) is False
    assert capsys.readouterr().out.startswith("Not yet:")


def test_train_arrays_message_uses_own_row_count(cases, df, capsys):
    _, bad = cases["check_train_arrays"]
    check_train_arrays(*bad)
    out = capsys.readouterr().out
    assert f"{len(df):,} rows" in out
    assert "split column" in out


def test_prediction_table_says_which_rows(cases, df, capsys):
    good, _ = cases["check_prediction_table"]
    table = good[0]
    check_prediction_table(table.iloc[:10], df, "baseline")
    assert "missing" in capsys.readouterr().out
    train_rows = df[df["split"] == "train"].head(3)
    extra = pd.concat(
        [
            table,
            pd.DataFrame(
                {
                    "accession": train_rows["accession"],
                    "species": train_rows["species"],
                    "y_true": train_rows["ec_class"],
                    "y_pred_baseline": 1,
                }
            ),
        ]
    )
    check_prediction_table(extra, df, "baseline")
    assert "not test proteins" in capsys.readouterr().out


def test_prediction_table_bad_columns_and_labels(cases, df, capsys):
    good, _ = cases["check_prediction_table"]
    table = good[0]
    assert check_prediction_table(table.drop(columns="y_pred_baseline"), df, "baseline") is False
    assert "y_pred_baseline" in capsys.readouterr().out
    assert check_prediction_table(table.assign(y_pred_baseline=9), df, "baseline") is False
    assert "1 to 7" in capsys.readouterr().out
    assert check_prediction_table(table, df, "finetuned") is False
    capsys.readouterr()


def test_prediction_table_accepts_dtype_differences(cases, df):
    good, _ = cases["check_prediction_table"]
    table = good[0].assign(y_pred_baseline=lambda t: t["y_pred_baseline"].astype("int32"))
    assert check_prediction_table(table, df, "baseline") is True


def test_counts_ignore_row_order_and_dtype(df, capsys):
    counts = df["species"].value_counts().astype("int32")
    assert check_species_counts(counts.sample(frac=1, random_state=1), df) is True
    capsys.readouterr()


def test_numeric_checks_reject_bool_and_nan(df, capsys):
    assert check_mean_length(True, df) is False
    assert check_mean_length(float("nan"), df) is False
    assert check_macro_f1(float("nan"), [1, 2], [1, 2]) is False
    capsys.readouterr()


def test_hyperparameter_messages(capsys):
    assert check_hyperparameters(3.0, 5e-5) is False
    assert "whole number" in capsys.readouterr().out
    assert check_hyperparameters(3, 5) is False
    assert "learning_rate" in capsys.readouterr().out
    assert check_hyperparameters(True, 5e-5) is False
    capsys.readouterr()


def test_saved_model_names_missing_files(tmp_path, capsys):
    (tmp_path / "config.json").write_text("x")
    assert check_saved_model(tmp_path) is False
    out = capsys.readouterr().out
    assert "model.safetensors" in out and "vocab.txt" in out
    assert check_saved_model(tmp_path / "nowhere") is False
    assert "folder" in capsys.readouterr().out


def test_saved_model_accepts_pytorch_bin(tmp_path, capsys):
    for name in ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"]:
        (tmp_path / name).write_text("x")
    assert check_saved_model(str(tmp_path)) is True
    capsys.readouterr()


def test_umap_rejects_nan(capsys):
    coords = np.zeros((4, 2))
    coords[0, 0] = np.nan
    assert check_umap_coords(coords, 4) is False
    assert "finite" in capsys.readouterr().out


def test_misclassified_wrong_url(cases, capsys):
    good, _ = cases["check_misclassified"]
    table = good[0].assign(uniprot_url="https://uniprot.org/x")
    assert check_misclassified(table, good[1]) is False
    assert "uniprot_url" in capsys.readouterr().out


def test_results_table_wrong_rows(results_fixture, preds_fixture, capsys):
    assert check_results_table(results_fixture.iloc[:5], preds_fixture) is False
    assert "rows" in capsys.readouterr().out


def test_fitted_classifier_without_all_classes(cases, capsys):
    X = np.random.default_rng(0).normal(size=(30, 3))
    clf = LogisticRegression().fit(X, np.arange(30) % 3 + 1)
    assert check_fitted_classifier(clf) is False
    assert "class" in capsys.readouterr().out


def test_messages_do_not_leak_expected_numbers(cases, capsys):
    """A wrong numeric answer must not print the expected value."""
    good, bad = cases["check_species_f1"]
    check_species_f1(*bad)
    out = capsys.readouterr().out
    assert f"{good[0]:.3f}" not in out and str(good[0])[:6] not in out
    assert re.search(r"\d", out) is not None  # echoes the intern's own value


@pytest.mark.parametrize(
    "name,expected",
    [
        ("check_ec_class_function", "first_ec_class"),
        ("check_species_counts", "species_counts"),
        ("check_mean_length", "yeast_mean_length"),
        ("check_majority_accuracy", "majority_accuracy"),
        ("check_fraction_truncated", "fraction_truncated"),
        ("check_multiclass_count", "n_multiclass"),
        ("check_macro_f1", "human_test_f1"),
        ("check_misclassified", "misclassified"),
        ("check_hyperparameters", "num_epochs"),
        ("check_saved_model", "path"),
    ],
)
def test_unfilled_names_notebook_variable(name, expected, cases, capsys):
    good, _ = cases[name]
    getattr(checks, name)(None, *good[1:])
    assert capsys.readouterr().out == f"Not yet: fill in {expected} first.\n"


def test_name_keyword_is_used_in_messages(cases, capsys):
    check_species_f1(None, cases["check_species_f1"][0][1], "yeast", "baseline", name="yeast_baseline_f1")
    assert capsys.readouterr().out == "Not yet: fill in yeast_baseline_f1 first.\n"
    check_umap_coords(None, 5, name="coords_after")
    assert capsys.readouterr().out == "Not yet: fill in coords_after first.\n"
    good, _ = cases["check_prediction_table"]
    check_prediction_table(None, good[1], "baseline", name="ft_preds")
    assert capsys.readouterr().out == "Not yet: fill in ft_preds first.\n"
    check_umap_coords(np.zeros((3, 3)), 3, name="coords_after")
    assert capsys.readouterr().out.startswith("Not yet: coords_after has shape")
    check_prediction_table(good[0].iloc[:5], good[1], "baseline", name="ft_preds")
    assert "ft_preds has 5 rows" in capsys.readouterr().out


def test_species_list_counts_problems(capsys):
    assert check_species_list(["human", "mouse", "fly", "yeast"]) is False
    assert capsys.readouterr().out == "Not yet: your list is missing 1 species.\n"
    assert check_species_list(SPECIES_ORDER + ["cat"]) is False
    assert "1 extra" in capsys.readouterr().out
