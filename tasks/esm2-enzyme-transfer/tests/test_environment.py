import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_core_imports():
    import torch, transformers, sklearn, pandas, numpy, matplotlib, umap, jupytext  # noqa: F401


def test_reference_and_outputs_are_ignored():
    for rel in ["tasks/esm2-enzyme-transfer/reference/x.py",
                "tasks/esm2-enzyme-transfer/outputs/x.npz"]:
        assert subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO).returncode == 0
