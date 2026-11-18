import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

RAW_COLUMNS = ["accession", "organism_id", "gene", "protein_name", "ec", "length", "sequence", "eggnog"]


@pytest.fixture
def synthetic_human():
    """2,000 human-like rows in 400 groups of 5, with classes 1-7 spread over groups."""
    rng = np.random.default_rng(42)
    group_class = np.arange(400) % 7 + 1
    group_ids = np.repeat([f"KOG{i:04d}" for i in range(400)], 5)
    ec_class = np.repeat(group_class, 5)
    order = rng.permutation(2000)
    return pd.DataFrame(
        {"accession": [f"P{i:05d}" for i in range(2000)], "ec_class": ec_class, "group_id": group_ids}
    ).iloc[order].reset_index(drop=True)


@pytest.fixture
def raw_fixture():
    """Small raw tables: GAPDH is multi-class, ENO1 has no eggNOG, mouse has one row.

    Filler human rows give the 20-fold split enough groups to run."""
    human = pd.DataFrame(
        [
            ["P04406", "9606", "GAPDH", "GAPDH", "1.2.1.12; 2.6.99.-", 335, "MGKVKV", "COG0057;KOG0657;"],
            ["P06733", "9606", "ENO1", "Enolase 1", "4.2.1.11", 434, "MSILKI", ""],
            ["P00533", "9606", "EGFR", "EGFR kinase", "2.7.10.1", 1210, "MRPSGT", "KOG1025;"],
            ["P12345", "9606", "AAT", "Aminotransferase", "2.6.1.1", 430, "MALLHS", "KOG1411;KOG9999;"],
        ],
        columns=RAW_COLUMNS,
    )
    filler = pd.DataFrame(
        {
            "accession": [f"Q{i:05d}" for i in range(180)],
            "organism_id": "9606",
            "gene": [f"G{i}" for i in range(180)],
            "protein_name": "filler",
            "ec": [f"{i % 6 + 1}.1.1.1" for i in range(180)],
            "length": 100,
            "sequence": "MAAAAA",
            "eggnog": [f"KOG{i + 5000};" for i in range(180)],
        }
    )
    human = pd.concat([human, filler], ignore_index=True)
    mouse = pd.DataFrame(
        [["P17182", "10090", "Eno1", "Enolase 1", "4.2.1.11", 434, "MSILKI", "KOG2670;"]],
        columns=RAW_COLUMNS,
    )
    empty = human.iloc[0:0]
    return {"human": human, "mouse": mouse, "fly": empty, "yeast": empty, "ecoli": empty}


@pytest.fixture
def preds_fixture():
    """Deterministic predictions for all five species (300 rows each).

    Yeast truth has no class 7. Floor predicts one class; fine-tuned is
    clearly better than baseline."""
    from helpers.data import SPECIES_ORDER

    rng = np.random.default_rng(7)
    frames = []
    for species in SPECIES_ORDER:
        n = 300
        max_class = 6 if species == "yeast" else 7
        y_true = rng.integers(1, max_class + 1, n)

        def noisy(p_correct):
            return np.where(rng.random(n) < p_correct, y_true, rng.integers(1, 8, n))

        frames.append(
            pd.DataFrame(
                {
                    "accession": [f"{species}_{i:04d}" for i in range(n)],
                    "species": species,
                    "y_true": y_true,
                    "y_pred_floor": 1,
                    "y_pred_baseline": noisy(0.45),
                    "y_pred_finetuned": noisy(0.85),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _cut(df, n_per_class, offset=0):
    """Take n_per_class rows per class, with sequences cut to 60 residues."""
    ranked = df.sort_values("accession")
    rank = ranked.groupby("ec_class").cumcount()
    picked = ranked[(rank >= offset) & (rank < offset + n_per_class)]
    picked = picked.assign(sequence=picked["sequence"].str.slice(0, 60))
    return picked.reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_df():
    """14 human rows (2 per class), sequences cut to 60 residues."""
    from helpers.data import load_dataset

    return _cut(load_dataset(), 2)


@pytest.fixture(scope="session")
def tiny_train_val():
    """(train_df, val_df): 2 train rows and 1 val row per class, sequences cut to 60 residues."""
    from helpers.data import load_dataset

    df = load_dataset()
    return _cut(df[df["split"] == "train"], 2), _cut(df[df["split"] == "val"], 1)
