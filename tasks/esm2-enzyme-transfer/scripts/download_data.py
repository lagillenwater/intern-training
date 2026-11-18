"""Download reviewed, EC-annotated proteins for each species from UniProt.

Run from the task directory: python -m scripts.download_data --out data
"""
import argparse
import datetime
import io
import json
from pathlib import Path

import pandas as pd
import requests

from helpers.data import SPECIES, SPECIES_ORDER

STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
QUERY_TEMPLATE = "reviewed:true AND organism_id:{taxon_id} AND ec:*"
FIELDS = "accession,organism_id,gene_primary,protein_name,ec,length,sequence,xref_eggnog"
COLUMNS = ["accession", "organism_id", "gene", "protein_name", "ec", "length", "sequence", "eggnog"]


def build_query(taxon_id: int) -> str:
    """Return the UniProt query for reviewed enzymes of one organism."""
    return QUERY_TEMPLATE.format(taxon_id=taxon_id)


def parse_uniprot_tsv(text: str) -> pd.DataFrame:
    """Parse UniProt TSV text into the standard columns; missing strings become ''."""
    df = pd.read_csv(io.StringIO(text), sep="\t", keep_default_na=False, dtype=str)
    df.columns = COLUMNS
    df["length"] = df["length"].astype(int)
    return df


def fetch_species(taxon_id: int, session: requests.Session | None = None) -> tuple[pd.DataFrame, str]:
    """Download one species; return the table and the UniProt release string."""
    session = session or requests.Session()
    params = {"query": build_query(taxon_id), "fields": FIELDS, "format": "tsv"}
    response = session.get(STREAM_URL, params=params, timeout=120)
    response.raise_for_status()
    return parse_uniprot_tsv(response.text), response.headers.get("X-UniProt-Release", "")


def main(out_dir: Path) -> None:
    """Write one raw TSV per species plus metadata.json under out_dir."""
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    counts, release = {}, ""
    for name in SPECIES_ORDER:
        df, release = fetch_species(SPECIES[name], session)
        df.to_csv(raw_dir / f"{name}.tsv", sep="\t", index=False)
        counts[name] = len(df)
    metadata = {
        "downloaded": datetime.date.today().isoformat(),
        "uniprot_release": release,
        "query_template": QUERY_TEMPLATE,
        "fields": FIELDS,
        "counts": counts,
    }
    (Path(out_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    main(parser.parse_args().out)
