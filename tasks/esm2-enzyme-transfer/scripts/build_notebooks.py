"""Build notebooks from the local reference sources.

Reads jupytext percent-format files from reference/notebooks/NN_*.py and writes
  - reference/build/NN_*.ipynb : full notebooks with solutions, for execution
  - notebooks/NN_*.ipynb       : public starters with solution blocks stripped

Run from the task directory: python -m scripts.build_notebooks
"""
import argparse
import re
from pathlib import Path

import jupytext
import nbformat

REPO_URL = "https://github.com/lagillenwater/intern-training"
PINNED = ("transformers", "accelerate", "umap-learn")
COLAB_GPU_PREFIXES = ("02", "03", "05")
COLAB_METADATA = {"accelerator": "GPU", "colab": {"gpuType": "T4"}}

BEGIN = "### BEGIN SOLUTION"
END = "### END SOLUTION"
STARTER = "### STARTER "
PLACEHOLDER = "# YOUR CODE HERE"

SETUP_TEMPLATE = '''# Setup: find the course materials and load our helper code. Run this cell first.
import subprocess
import sys
from pathlib import Path

# Colab has a module called google.colab; a laptop does not.
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # On Colab: download the course materials once.
    TASK_DIR = Path("/content/intern-training/tasks/esm2-enzyme-transfer")
    if not TASK_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "{materials_ref}",
             "{repo_url}", "/content/intern-training"],
            check=True,
        )
    # Install the exact package versions this course was tested with.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", {pins}],
        check=True,
    )
else:
    # On a laptop: walk up from the current folder until we find the folder
    # that holds both helpers/ and checks.py.
    TASK_DIR = None
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / "helpers").is_dir() and (folder / "checks.py").exists():
            TASK_DIR = folder
            break
    if TASK_DIR is None:
        raise RuntimeError("Could not find the course materials (a folder with helpers/ and checks.py).")

# Make our helper code importable, then load it.
sys.path.insert(0, str(TASK_DIR))
from helpers import *
from checks import *

print(f"Setup complete. Materials: {{TASK_DIR}}")
'''


def read_pins(requirements: Path) -> list:
    """Return the pinned `name==version` lines for the packages we install on Colab."""
    lines = requirements.read_text().splitlines()
    pins = [ln.strip() for ln in lines if re.split(r"==", ln.strip())[0] in PINNED]
    if len(pins) != len(PINNED):
        raise ValueError(f"expected pins for {PINNED} in {requirements}, found {pins}")
    return pins


def setup_cell(materials_ref: str) -> str:
    """Return the source of the first code cell of every notebook."""
    pins = read_pins(Path(__file__).resolve().parent.parent / "requirements.txt")
    pin_args = ", ".join(repr(p) for p in pins)
    return SETUP_TEMPLATE.format(materials_ref=materials_ref, repo_url=REPO_URL, pins=pin_args)


def strip_solutions(source: str) -> str:
    """Replace each solution block with its STARTER lines, or a placeholder if it has none."""
    out, starters, inside = [], [], False
    for number, line in enumerate(source.splitlines(), start=1):
        marker = line.strip()
        if marker == BEGIN:
            if inside:
                raise ValueError(f"line {number}: nested {BEGIN}")
            inside, starters = True, []
        elif marker == END:
            if not inside:
                raise ValueError(f"line {number}: {END} without {BEGIN}")
            out.extend(starters or [PLACEHOLDER])
            inside = False
        elif inside:
            if line.lstrip().startswith(STARTER):
                indent = line[: len(line) - len(line.lstrip())]
                starters.append(indent + line.lstrip()[len(STARTER):])
        elif line.lstrip().startswith(STARTER.strip()):
            raise ValueError(f"line {number}: STARTER outside a solution block")
        else:
            out.append(line)
    if inside:
        raise ValueError(f"{BEGIN} was never closed")
    return "\n".join(out) + "\n"


def build_one(source: Path, out: Path, materials_ref: str, strip: bool) -> None:
    """Convert one percent-format .py to an .ipynb with the setup cell prepended and no outputs."""
    text = source.read_text()
    if strip:
        text = strip_solutions(text)
    nb = jupytext.reads(text, fmt="py:percent")
    nb.cells.insert(0, nbformat.v4.new_code_cell(setup_cell(materials_ref)))
    for index, cell in enumerate(nb.cells):
        cell.id = f"cell-{index}"  # deterministic ids keep rebuilds byte-identical
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    # Keep metadata minimal so diffs stay clean.
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if source.name.startswith(COLAB_GPU_PREFIXES):
        nb.metadata.update(COLAB_METADATA)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)


def build_single(source: Path, materials_ref: str = "esm2-v1") -> Path:
    """Build one percent-format .py (any path) into SOURCE.ipynb next to it, with no stripping."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"--only source not found: {source}")
    out = source.with_suffix(".ipynb")
    build_one(source, out, materials_ref, strip=False)
    print(f"Built {out}")
    return out


def build(reference_dir: Path, starter_dir: Path, build_dir: Path, materials_ref: str = "esm2-v1") -> None:
    """Build every reference/notebooks/NN_*.py into full and starter notebooks."""
    source_dir = reference_dir / "notebooks"
    if not reference_dir.exists() or not source_dir.exists():
        print(f"Reference directory not found: {reference_dir}; nothing to build.")
        return
    sources = sorted(source_dir.glob("[0-9][0-9]_*.py"))
    for source in sources:
        build_one(source, build_dir / f"{source.stem}.ipynb", materials_ref, strip=False)
        build_one(source, starter_dir / f"{source.stem}.ipynb", materials_ref, strip=True)
        print(f"Built {source.name}")


def main() -> None:
    """Command-line entry point."""
    task = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--materials-ref", default="esm2-v1", help="git branch or tag Colab clones")
    parser.add_argument("--only", type=Path, help="build only this percent-format .py into SOURCE.ipynb next to it (no stripping)")
    args = parser.parse_args()
    if args.only is not None:
        build_single(args.only, args.materials_ref)
        return
    build(task / "reference", task / "notebooks", task / "reference" / "build", args.materials_ref)


if __name__ == "__main__":
    main()
