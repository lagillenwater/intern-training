import pytest

from scripts.download_data import build_query, fetch_species, parse_uniprot_tsv

FIXTURE = ("Entry\tOrganism (ID)\tGene Names (primary)\tProtein names\tEC number\tLength\tSequence\teggNOG\n"
           "P04406\t9606\tGAPDH\tGlyceraldehyde-3-phosphate dehydrogenase\t1.2.1.12; 2.6.99.-\t335\tMGKV\tKOG0657;\n"
           "P06733\t9606\tENO1\tAlpha-enolase\t4.2.1.11\t434\tMSIL\t\n")


def test_build_query():
    assert build_query(9606) == "reviewed:true AND organism_id:9606 AND ec:*"


def test_parse_uniprot_tsv():
    df = parse_uniprot_tsv(FIXTURE)
    assert list(df.columns) == ["accession", "organism_id", "gene", "protein_name", "ec", "length", "sequence", "eggnog"]
    assert df.loc[0, "ec"] == "1.2.1.12; 2.6.99.-"
    assert df["length"].dtype.kind == "i"
    assert df.loc[1, "eggnog"] == ""


@pytest.mark.network
def test_fetch_species_ecoli():
    df, release = fetch_species(83333)
    assert len(df) > 1500 and release
