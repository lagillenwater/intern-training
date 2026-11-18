"""Work-directory and missing-file helpers that every notebook calls first."""
import os
import sys
from pathlib import Path

from helpers.data import task_dir

DRIVE_ROOT = Path("/content/drive/MyDrive/esm2-enzyme-transfer")


def in_colab() -> bool:
    """Return True when running inside Google Colab."""
    return "google.colab" in sys.modules


def setup_workdir(run_name: str = "esm2_8M") -> Path:
    """Create and return the folder for this run, with a figures/ subfolder.

    Uses $ENZYME_WORK_DIR/<run_name> if that variable is set. On Colab it mounts
    Google Drive so files survive after the session ends. Otherwise it uses
    outputs/<run_name> inside the task folder.
    """
    env = os.environ.get("ENZYME_WORK_DIR")
    if env:
        base = Path(env)
    elif in_colab():
        from google.colab import drive

        drive.mount("/content/drive")
        base = DRIVE_ROOT
    else:
        base = task_dir() / "outputs"
    workdir = base / run_name
    (workdir / "figures").mkdir(parents=True, exist_ok=True)
    return workdir


def require_file(path: Path, made_by: str) -> Path:
    """Return path if it exists, otherwise raise an error that says which notebook makes it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. It is made by {made_by}. "
            "Run that notebook first, or ask your mentor for the catch-up files."
        )
    return path
