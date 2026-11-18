"""Checks on the intern-facing README and AI-use log.

The README must have the sections the intern relies on, and must never show a
reference score. The score check reads the local reference results, which only
exist on the mentor's machine after the reference run, so it skips until then.
"""

import re

import pandas as pd
import pytest

from helpers import task_dir

AI_LOG_HEADER = "| Date | Notebook | What I asked | What I kept | What I changed or rejected |"


def test_readme_sections():
    text = (task_dir() / "README.md").read_text()
    for heading in ["Before day 1", "Schedule", "AI use", "Deliverables", "Troubleshooting"]:
        assert f"## {heading}" in text
    assert "Use this template" in text and "AI_LOG.md" in text


def test_readme_shows_no_reference_scores():
    text = (task_dir() / "README.md").read_text()
    results_path = task_dir() / "reference/outputs/esm2_8M/results.tsv"
    if not results_path.exists():
        pytest.skip("reference results not built yet: reference/outputs/esm2_8M/results.tsv")
    results = pd.read_csv(results_path, sep="\t")
    for v in results.query("model != 'floor'").macro_f1:
        assert not re.search(rf"\b{v:.2f}\b", text)


def test_ai_log_template():
    text = (task_dir() / "AI_LOG.md").read_text()
    lines = text.splitlines()
    assert AI_LOG_HEADER in lines
    rows = lines[lines.index(AI_LOG_HEADER) + 2 :]
    assert any(row.startswith("|") for row in rows), "AI_LOG.md needs one example row"
