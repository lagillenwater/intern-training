import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_core_imports():
    import torch, transformers, sklearn, pandas, numpy, matplotlib, umap, jupytext  # noqa: F401


def test_reference_and_outputs_are_ignored():
    for rel in ["reference/x.py", "outputs/x.npz"]:
        assert subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO).returncode == 0
