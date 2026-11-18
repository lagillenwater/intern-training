"""Check functions the notebooks call after each TODO cell.

Each check returns True or False and prints exactly one line, starting with
"Check passed:" or "Not yet:". Messages say what to look at and never reveal
the expected value of a computed result. A check never raises: a problem in
the intern's input (wrong type, wrong shape, a function that crashes) is
reported as "Not yet: ...".
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted

from helpers import MAX_RESIDUES, SPECIES_ORDER, ec_classes

__all__ = [
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

CLASS_LABELS = list(range(1, 8))
MODELS = ("floor", "baseline", "finetuned")
UNIPROT_URL = "https://www.uniprot.org/uniprotkb/"


# --- private plumbing -------------------------------------------------------


def _report(ok: bool, passed: str, hint: str) -> bool:
    """Print one line (passed message or hint) and return ok."""
    print(f"Check passed: {passed}" if ok else f"Not yet: {hint}")
    return bool(ok)


def _unfilled(value, name: str) -> bool:
    """If value is None or Ellipsis, print the fill-in message and return True."""
    if value is None or value is Ellipsis:
        print(f"Not yet: fill in {name} first.")
        return True
    return False


def _safe(func):
    """Turn any unexpected error inside a check into a one-line Not yet."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:  # noqa: BLE001  (a check must never show a traceback)
            kind = type(err).__name__
            text = " ".join(str(err).split())[:100]
            return _report(False, "", f"this check could not run on your value ({kind}: {text}). Look at what you passed in.")

    return wrapper


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_number(value, name: str, expected: float, tol: float, passed: str, hint: str) -> bool:
    """Compare a single number to a recomputed one without revealing the answer."""
    if not _is_number(value):
        return _report(False, "", f"{name} should be a single number, but it is a {type(value).__name__}.")
    if not np.isfinite(value):
        return _report(False, "", f"{name} is {value}, which is not a usable number. {hint}")
    ok = abs(float(value) - float(expected)) <= tol
    return _report(ok, passed, f"{name} is {float(value):.4g}, which does not match the data. {hint}")


def _same_table(got: pd.DataFrame, want: pd.DataFrame) -> bool:
    """Equal values after sorting rows and columns; dtype differences are ignored."""
    if got.shape != want.shape or set(got.columns) != set(want.columns):
        return False
    got = got[want.columns].sort_index()
    want = want.sort_index()
    return bool(set(got.index) == set(want.index) and np.array_equal(got.to_numpy(dtype=float), want.to_numpy(dtype=float)))


# --- notebook 00 ------------------------------------------------------------


@_safe
def check_species_list(species) -> bool:
    if _unfilled(species, "species"):
        return False
    if not isinstance(species, list):
        return _report(False, "", f"species should be a list (square brackets), but it is a {type(species).__name__}.")
    missing = [s for s in SPECIES_ORDER if s not in species]
    extra = [s for s in species if s not in SPECIES_ORDER]
    if missing:
        return _report(False, "", f"your list is missing {len(missing)} species.")
    if extra:
        return _report(False, "", f"your list has {len(extra)} extra item(s) that are not one of the species.")
    if len(species) != len(set(species)):
        return _report(False, "", "your list repeats a species.")
    return _report(True, "your list has all five species.", "")


@_safe
def check_ec_class_function(first_ec_class) -> bool:
    if _unfilled(first_ec_class, "first_ec_class"):
        return False
    if not callable(first_ec_class):
        return _report(False, "", "first_ec_class should be a function (made with def), but it is not callable.")
    for example, want in [("3.4.21.4", 3), ("2.7.11.1", 2), ("1.2.1.12; 1.1.1.27", 1), ("3.4.-.-", 3)]:
        try:
            got = first_ec_class(example)
        except Exception as err:  # noqa: BLE001
            text = " ".join(str(err).split())[:80]
            return _report(False, "", f'your function raised {type(err).__name__}: {text} for input "{example}".')
        if not _is_int(got):
            return _report(False, "", f'your function returned a {type(got).__name__} for input "{example}"; it should return an int.')
        if int(got) != want:
            return _report(False, "", f'your function returned {int(got)} for input "{example}", which is not the right class.')
    return _report(True, "your function finds the right class for all four examples.", "")


@_safe
def check_species_counts(species_counts, df) -> bool:
    if _unfilled(species_counts, "species_counts"):
        return False
    if not isinstance(species_counts, pd.Series):
        return _report(False, "", f"species_counts should be a pandas Series (what value_counts() gives), but it is a {type(species_counts).__name__}.")
    want = df["species"].value_counts()
    ok = set(species_counts.index) == set(want.index) and bool((species_counts.reindex(want.index).to_numpy() == want.to_numpy()).all())
    return _report(ok, "your counts match the table.", "species_counts does not match the table. Did you count the species column of df?")


@_safe
def check_mean_length(yeast_mean_length, df, species: str = "yeast") -> bool:
    if _unfilled(yeast_mean_length, "yeast_mean_length"):
        return False
    want = df.loc[df["species"] == species, "length"].mean()
    return _check_number(
        yeast_mean_length, "yeast_mean_length", want, 1e-6, f"the mean length for {species} matches.",
        f"Did you keep only the {species} rows and take the mean of the length column?",
    )


# --- notebook 01 ------------------------------------------------------------


@_safe
def check_class_counts(counts, df) -> bool:
    if _unfilled(counts, "counts"):
        return False
    if not isinstance(counts, pd.DataFrame):
        return _report(False, "", f"counts should be a table (DataFrame), but it is a {type(counts).__name__}.")
    want = pd.crosstab(df["species"], df["ec_class"])
    if counts.shape != want.shape:
        return _report(False, "", f"counts has {counts.shape[0]} rows and {counts.shape[1]} columns; it should have one row per species and one column per class.")
    ok = _same_table(counts, want)
    return _report(ok, "your table of class counts matches.", "the numbers in counts do not match. Did you cross-tabulate species against ec_class?")


@_safe
def check_majority_accuracy(majority_accuracy, df) -> bool:
    if _unfilled(majority_accuracy, "majority_accuracy"):
        return False
    train = df[(df["species"] == "human") & (df["split"] == "train")]
    want = train["ec_class"].value_counts(normalize=True).max()
    return _check_number(
        majority_accuracy, "majority_accuracy", want, 1e-6, "your majority-class accuracy is right.",
        "It should be a fraction between 0 and 1: the share of human training proteins in their most common class.",
    )


@_safe
def check_fraction_truncated(fraction_truncated, df) -> bool:
    if _unfilled(fraction_truncated, "fraction_truncated"):
        return False
    want = (df["length"] > MAX_RESIDUES).mean()
    return _check_number(
        fraction_truncated, "fraction_truncated", want, 1e-6, "your fraction of too-long proteins is right.",
        "It should be a fraction between 0 and 1: proteins longer than the model's limit divided by all proteins.",
    )


@_safe
def check_multiclass_count(n_multiclass, raw_human) -> bool:
    if _unfilled(n_multiclass, "n_multiclass"):
        return False
    want = int(raw_human["ec"].fillna("").map(lambda ec: len(ec_classes(ec)) > 1).sum())
    if not _is_number(n_multiclass):
        return _report(False, "", f"n_multiclass should be a single number, but it is a {type(n_multiclass).__name__}.")
    return _report(
        int(n_multiclass) == want and float(n_multiclass) == int(n_multiclass), "your count of multi-class proteins is right.",
        f"n_multiclass is {n_multiclass:,}, which does not match. Count the rows whose ec column lists more than one top-level class.",
    )


# --- notebook 02 ------------------------------------------------------------


@_safe
def check_train_arrays(X_train, y_train, X, df) -> bool:
    if _unfilled(X_train, "X_train") or _unfilled(y_train, "y_train"):
        return False
    mask = (df["split"] == "train").to_numpy()
    want_X, want_y = np.asarray(X)[mask], df.loc[mask, "ec_class"].to_numpy()
    got_X, got_y = np.asarray(X_train), np.asarray(y_train)
    if got_X.shape[0] != want_X.shape[0]:
        return _report(False, "", f"X_train has {got_X.shape[0]:,} rows but should only have the training proteins. Did you use the split column?")
    if got_y.shape[0] != want_y.shape[0]:
        return _report(False, "", f"y_train has {got_y.shape[0]:,} labels but should only have the training proteins. Did you use the split column?")
    if got_X.shape != want_X.shape or not np.allclose(got_X, want_X):
        return _report(False, "", 'X_train has the right number of rows, but they are not the training rows of X. Select them with df["split"] == "train".')
    if not np.array_equal(got_y, want_y):
        return _report(False, "", 'y_train has the right length, but the labels are not the training labels. Take the ec_class column where df["split"] == "train".')
    return _report(True, "X_train and y_train hold the training proteins.", "")


@_safe
def check_fitted_classifier(clf) -> bool:
    if _unfilled(clf, "clf"):
        return False
    if not hasattr(clf, "predict"):
        return _report(False, "", "clf does not have a predict method. Did you create a classifier such as LogisticRegression?")
    try:
        check_is_fitted(clf)
    except NotFittedError:
        return _report(False, "", "clf has not been trained yet. Call clf.fit(X_train, y_train).")
    classes = [int(c) for c in getattr(clf, "classes_", [])]
    if classes != CLASS_LABELS:
        return _report(False, "", f"clf has learned {len(classes)} classes but should know all seven. Did you train on the training proteins of every class?")
    return _report(True, "clf is trained and knows all seven classes.", "")


# --- notebooks 02 and 04 ----------------------------------------------------


@_safe
def check_macro_f1(human_test_f1, y_true, y_pred) -> bool:
    if _unfilled(human_test_f1, "human_test_f1"):
        return False
    want = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return _check_number(
        human_test_f1, "human_test_f1", want, 1e-9, "your macro-F1 is right.",
        'Use f1_score(y_true, y_pred, average="macro").',
    )


# --- notebooks 02 and 03 ----------------------------------------------------


@_safe
def check_prediction_table(preds, df, model, *, name="preds") -> bool:
    if _unfilled(preds, name):
        return False
    if not isinstance(preds, pd.DataFrame):
        return _report(False, "", f"{name} should be a table (DataFrame), but it is a {type(preds).__name__}.")
    pred_col = f"y_pred_{model}"
    absent = [c for c in ["accession", "species", "y_true", pred_col] if c not in preds.columns]
    if absent:
        return _report(False, "", f"{name} is missing the column(s): {', '.join(absent)}.")
    test = df[df["split"] == "test"]
    got, want = preds["accession"], set(test["accession"])
    if got.duplicated().any():
        return _report(False, "", f"{name} lists some proteins more than once.")
    extra = set(got) - want
    if extra:
        kind = "not test proteins" if extra & set(df["accession"]) else "not in the dataset at all"
        return _report(False, "", f"{name} has {len(preds):,} rows, and some of them are {kind}. Keep only the rows where split is test.")
    if want - set(got):
        return _report(False, "", f"{name} has {len(preds):,} rows and is missing some test proteins. Predict every protein where split is test.")
    values = pd.to_numeric(preds[pred_col], errors="coerce")
    if values.isna().any() or not values.isin(CLASS_LABELS).all():
        return _report(False, "", f"{pred_col} should hold class numbers from 1 to 7 only.")
    return _report(True, "your prediction table covers the test proteins with valid classes.", "")


# --- notebook 03 ------------------------------------------------------------


@_safe
def check_hyperparameters(num_epochs, learning_rate) -> bool:
    if _unfilled(num_epochs, "num_epochs") or _unfilled(learning_rate, "learning_rate"):
        return False
    if not _is_int(num_epochs):
        return _report(False, "", f"num_epochs should be a whole number, but you have {num_epochs!r}.")
    if not 1 <= num_epochs <= 10:
        return _report(False, "", f"num_epochs is {num_epochs:,}; choose a whole number from 1 to 10.")
    if not _is_number(learning_rate):
        return _report(False, "", f"learning_rate should be a number like 5e-5, but you have {learning_rate!r}.")
    if not 1e-6 <= learning_rate <= 1e-3:
        return _report(False, "", f"learning_rate is {learning_rate:g}; choose a number from 0.000001 (1e-6) to 0.001 (1e-3).")
    return _report(True, "your training settings are in a sensible range.", "")


@_safe
def check_saved_model(path) -> bool:
    if _unfilled(path, "path"):
        return False
    folder = Path(path)
    if not folder.is_dir():
        return _report(False, "", f"there is no folder at {folder}. Did trainer.save_model(path) run?")
    have = {p.name for p in folder.iterdir()}
    missing = [n for n in ["config.json", "tokenizer_config.json", "vocab.txt"] if n not in have]
    if not have & {"model.safetensors", "pytorch_model.bin"}:
        missing.append("model.safetensors")
    return _report(not missing, "the saved model folder has its settings, weights, and tokenizer.", f"the folder is missing: {', '.join(missing)}. Run trainer.save_model(path) again.")


# --- notebook 04 ------------------------------------------------------------


@_safe
def check_species_f1(value, preds, species, model, *, name="the yeast macro-F1") -> bool:
    if _unfilled(value, name):
        return False
    rows = preds[preds["species"] == species]
    want = f1_score(rows["y_true"], rows[f"y_pred_{model}"], average="macro", zero_division=0)
    return _check_number(
        value, name, want, 1e-9, f"your {species} macro-F1 for {model} is right.",
        f"Score only the {species} rows, using y_true and y_pred_{model}.",
    )


@_safe
def check_results_table(results, preds) -> bool:
    if _unfilled(results, "results"):
        return False
    if not isinstance(results, pd.DataFrame):
        return _report(False, "", f"results should be a table (DataFrame), but it is a {type(results).__name__}.")
    cols = ["species", "model", "macro_f1", "accuracy", "ci_low", "ci_high"]
    absent = [c for c in cols if c not in results.columns]
    if absent:
        return _report(False, "", f"results is missing the column(s): {', '.join(absent)}. Did you build it with score_table?")
    if len(results) != len(SPECIES_ORDER) * len(MODELS):
        return _report(False, "", f"results has {len(results):,} rows, but it needs one row for every species and model pair.")
    wide = results.pivot_table(index="species", columns="model", values="macro_f1", aggfunc="first")
    got = wide.stack()
    truth = preds.melt(id_vars=["species", "y_true"], value_vars=[f"y_pred_{m}" for m in MODELS], var_name="model", value_name="y_pred")
    truth["model"] = truth["model"].str.removeprefix("y_pred_")
    want = truth.groupby(["species", "model"]).apply(
        lambda g: f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0), include_groups=False
    )
    aligned = got.reindex(want.index)
    ok = len(got) == len(want) and aligned.notna().all() and bool((np.abs(aligned - want) <= 1e-9).all())
    return _report(ok, "your results table has every species and model, and the macro-F1 values match.", "some macro_f1 values do not match your predictions. Rebuild the table with score_table(preds).")


# --- notebook 05 ------------------------------------------------------------


@_safe
def check_umap_coords(coords, n, *, name="coords") -> bool:
    if _unfilled(coords, name):
        return False
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return _report(False, "", f"{name} has shape {arr.shape}; it should have two columns, one for x and one for y.")
    if arr.shape[0] != n:
        return _report(False, "", f"{name} has {arr.shape[0]:,} rows but should have one per protein ({n:,}).")
    if not np.isfinite(arr.astype(float)).all():
        return _report(False, "", f"{name} has values that are not finite (NaN or infinity).")
    return _report(True, f"{name} has one finite x, y point per protein.", "")


@_safe
def check_misclassified(misclassified, preds, species=("yeast", "ecoli")) -> bool:
    if _unfilled(misclassified, "misclassified"):
        return False
    if not isinstance(misclassified, pd.DataFrame):
        return _report(False, "", f"misclassified should be a table (DataFrame), but it is a {type(misclassified).__name__}.")
    cols = ["accession", "species", "y_true", "y_pred_finetuned", "uniprot_url"]
    absent = [c for c in cols if c not in misclassified.columns]
    if absent:
        return _report(False, "", f"misclassified is missing the column(s): {', '.join(absent)}.")
    names = [species] if isinstance(species, str) else list(species)
    errors = preds[preds["species"].isin(names) & (preds["y_true"] != preds["y_pred_finetuned"])]
    if misclassified["accession"].duplicated().any() or set(misclassified["accession"]) != set(errors["accession"]):
        return _report(False, "", f"misclassified has {len(misclassified):,} rows; it should list exactly the proteins the fine-tuned model got wrong in {' and '.join(names)}.")
    merged = misclassified.merge(errors, on="accession", suffixes=("", "_true"))
    if not ((merged["y_true"] == merged["y_true_true"]) & (merged["y_pred_finetuned"] == merged["y_pred_finetuned_true"])).all():
        return _report(False, "", "the y_true or y_pred_finetuned values in misclassified do not match preds for those proteins.")
    if not (misclassified["uniprot_url"] == UNIPROT_URL + misclassified["accession"].astype(str)).all():
        return _report(False, "", f"uniprot_url should be {UNIPROT_URL} followed by the accession.")
    return _report(True, "misclassified lists the fine-tuned errors with working UniProt links.", "")
