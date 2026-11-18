import pandas as pd

from helpers.data import (
    SPECIES_ORDER,
    assign_human_split,
    build_dataset,
    ec_classes,
    eggnog_ids,
    load_dataset,
    load_divergence,
)


def test_ec_classes():
    assert ec_classes("1.2.1.12; 2.6.99.-") == {1, 2}
    assert ec_classes("3.4.-.-") == {3}
    assert ec_classes("2.7.11.1; 2.7.10.2") == {2}
    assert ec_classes("") == set()


def test_eggnog_ids():
    assert eggnog_ids("KOG0657;") == ["KOG0657"]
    assert eggnog_ids("COG0057;KOG0657;") == ["COG0057", "KOG0657"]
    assert eggnog_ids("") == []


def test_assign_human_split(synthetic_human):
    split = assign_human_split(synthetic_human, seed=0)
    assert set(split) == {"train", "val", "test"}
    assert (synthetic_human.assign(s=split).groupby("group_id")["s"].nunique() == 1).all()
    frac = split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["val"] - 0.15) < 0.03
    assert (pd.crosstab(synthetic_human["ec_class"], split) > 0).all().all()
    assert (assign_human_split(synthetic_human, seed=0) == split).all()


def test_build_dataset_drops_multiclass(raw_fixture):
    df = build_dataset(raw_fixture)
    assert "P04406" not in set(df.accession)
    assert df.loc[df.accession == "P06733", "group_id"].item() == "P06733"
    assert (df.loc[df.species != "human", "split"] == "test").all()
    assert df["ec_class"].dtype.kind == "i"


def test_committed_dataset():
    df = load_dataset()
    human = df[df.species == "human"]
    assert list(dict.fromkeys(df.species)) == SPECIES_ORDER
    assert df.ec_class.between(1, 7).all()
    assert 4300 < len(human) < 4616
    assert (human.groupby("group_id")["split"].nunique() == 1).all()
    assert (pd.crosstab(human.ec_class, human.split) > 0).all().all()


def test_divergence_table():
    div = load_divergence()
    assert list(div.species) == SPECIES_ORDER and div.mya.is_monotonic_increasing
