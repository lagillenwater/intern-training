"""Build data/enzymes.tsv from the raw UniProt tables.

Run from the task directory: python -m scripts.build_dataset
"""
import pandas as pd

from helpers.data import SPECIES_ORDER, build_dataset, load_raw, task_dir


def main() -> None:
    """Write data/enzymes.tsv and print per-species and per-split counts."""
    raw = {name: load_raw(name) for name in SPECIES_ORDER}
    df = build_dataset(raw)
    df.to_csv(task_dir() / "data" / "enzymes.tsv", sep="\t", index=False)
    raw_counts = pd.Series({name: len(raw[name]) for name in SPECIES_ORDER})
    kept = df["species"].value_counts().reindex(SPECIES_ORDER)
    print("Rows per species (raw, kept, dropped as multi-class):")
    print(pd.DataFrame({"raw": raw_counts, "kept": kept, "dropped": raw_counts - kept}))
    print("\nHuman rows per split:")
    print(df.loc[df.species == "human", "split"].value_counts())
    print("\nHuman class by split:")
    print(pd.crosstab(df.loc[df.species == "human", "ec_class"], df.loc[df.species == "human", "split"]))


if __name__ == "__main__":
    main()
