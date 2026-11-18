"""Dataset definitions, labels, grouped split, and loaders shared by scripts and notebooks."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPECIES = {"human": 9606, "mouse": 10090, "fly": 7227, "yeast": 559292, "ecoli": 83333}
SPECIES_ORDER = ["human", "mouse", "fly", "yeast", "ecoli"]
CLASS_NAMES = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
    7: "Translocases",
}
MAX_RESIDUES = 1022

DATASET_COLUMNS = [
    "accession", "species", "gene", "protein_name", "ec", "ec_class",
    "length", "sequence", "eggnog", "group_id", "split",
]
N_FOLDS = 20
TRAIN_FOLDS = 14  # folds 0-13 train, 14-16 val, 17-19 test
VAL_FOLDS = 17


def task_dir() -> Path:
    """Return the task folder (the one holding data/, helpers/, and scripts/)."""
    return Path(__file__).resolve().parent.parent


def ec_classes(ec: str) -> set[int]:
    """Return the set of top-level EC classes in a semicolon-separated EC string."""
    return {int(part.strip().split(".")[0]) for part in ec.split(";") if part.strip()}


def eggnog_ids(field: str) -> list[str]:
    """Split a semicolon-separated eggNOG field into a list of IDs."""
    return [part for part in field.split(";") if part]


def assign_human_split(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Assign train/val/test (70/15/15) to human rows, keeping each group_id together."""
    folds = np.empty(len(df), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(df, df["ec_class"], df["group_id"])):
        folds[test_idx] = fold
    labels = np.array(["train", "val", "test"])
    return pd.Series(labels[(folds >= TRAIN_FOLDS).astype(int) + (folds >= VAL_FOLDS)], index=df.index)


def build_dataset(raw: dict[str, pd.DataFrame], seed: int = 0) -> pd.DataFrame:
    """Combine raw species tables into the labeled dataset with split and group columns."""
    parts = [raw[name].assign(species=name) for name in SPECIES_ORDER]
    df = pd.concat(parts, ignore_index=True).drop(columns="organism_id")
    classes = df["ec"].map(ec_classes)
    df = df[classes.map(len) == 1].copy()
    df["ec_class"] = classes[df.index].map(lambda s: next(iter(s))).astype(int)
    first_id = df["eggnog"].map(lambda f: (eggnog_ids(f) or [""])[0])
    df["group_id"] = first_id.where(first_id != "", df["accession"])
    df["split"] = "test"
    is_human = df["species"] == "human"
    df.loc[is_human, "split"] = assign_human_split(df[is_human], seed)
    return df[DATASET_COLUMNS].reset_index(drop=True)


def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    for col in ("length", "ec_class"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the committed enzyme table (data/enzymes.tsv by default)."""
    return _read_tsv(path or task_dir() / "data" / "enzymes.tsv")


def load_raw(species: str) -> pd.DataFrame:
    """Load the raw UniProt table for one species."""
    return _read_tsv(task_dir() / "data" / "raw" / f"{species}.tsv")


def load_divergence() -> pd.DataFrame:
    """Load divergence times from human in millions of years (columns species, mya, source)."""
    df = pd.read_csv(task_dir() / "data" / "divergence_times.tsv", sep="\t", keep_default_na=False)
    return df
