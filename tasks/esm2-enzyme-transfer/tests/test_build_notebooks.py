import nbformat
import pytest

from helpers import task_dir
from scripts.build_notebooks import build, build_single, setup_cell, strip_solutions

SRC = """# %%
# TODO: fit the classifier
# Hint: use LogisticRegression
### BEGIN SOLUTION
### STARTER clf = None  # replace None with your code
clf = LogisticRegression().fit(X, y)
### END SOLUTION
check_fitted_classifier(clf)
"""


def test_strip_keeps_todo_and_starter():
    out = strip_solutions(SRC)
    assert "# TODO: fit the classifier" in out and "clf = None  # replace None with your code" in out
    assert "LogisticRegression().fit" not in out
    assert "SOLUTION" not in out and "STARTER" not in out


def test_strip_placeholder():
    assert "# YOUR CODE HERE" in strip_solutions("### BEGIN SOLUTION\nx = 1\n### END SOLUTION\n")


def test_unbalanced_markers():
    with pytest.raises(ValueError):
        strip_solutions("### BEGIN SOLUTION\nx = 1\n")


def test_nested_and_stray_markers():
    with pytest.raises(ValueError):
        strip_solutions("### BEGIN SOLUTION\n### BEGIN SOLUTION\n### END SOLUTION\n")
    with pytest.raises(ValueError):
        strip_solutions("x = 1\n### END SOLUTION\n")


def test_setup_cell_uses_pins_and_ref():
    cell = setup_cell("testref")
    assert "testref" in cell
    assert "transformers==5.23.0" in cell and "umap-learn==0.6.0" in cell
    compile(cell, "setup", "exec")


def test_build(tmp_path):
    (tmp_path / "ref/notebooks").mkdir(parents=True)
    (tmp_path / "ref/notebooks/02_baseline.py").write_text("# %%\n" + SRC)
    build(tmp_path / "ref", tmp_path / "starter", tmp_path / "build", materials_ref="testref")
    nb = nbformat.read(tmp_path / "starter/02_baseline.ipynb", as_version=4)
    assert "testref" in nb.cells[0].source
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    assert all(c.get("execution_count") is None for c in nb.cells if c.cell_type == "code")
    assert nb.metadata["accelerator"] == "GPU"
    assert "LogisticRegression().fit" in (tmp_path / "build/02_baseline.ipynb").read_text()


def test_build_missing_reference_dir(tmp_path, capsys):
    build(tmp_path / "nope", tmp_path / "starter", tmp_path / "build")
    assert "not found" in capsys.readouterr().out
    assert not (tmp_path / "starter").exists()


def test_no_colab_metadata_for_other_notebooks(tmp_path):
    (tmp_path / "ref/notebooks").mkdir(parents=True)
    (tmp_path / "ref/notebooks/01_look.py").write_text("# %%\nx = 1\n")
    build(tmp_path / "ref", tmp_path / "starter", tmp_path / "build")
    nb = nbformat.read(tmp_path / "starter/01_look.ipynb", as_version=4)
    assert "accelerator" not in nb.metadata


def test_build_twice_is_identical(tmp_path):
    (tmp_path / "ref/notebooks").mkdir(parents=True)
    (tmp_path / "ref/notebooks/02_baseline.py").write_text("# %%\n" + SRC)
    texts = []
    for _ in range(2):
        build(tmp_path / "ref", tmp_path / "starter", tmp_path / "build")
        texts.append((tmp_path / "starter/02_baseline.ipynb").read_text())
    assert texts[0] == texts[1]


def test_build_single(tmp_path):
    src = tmp_path / "tools" / "colab_timing.py"
    src.parent.mkdir()
    src.write_text("# %%\n" + SRC)
    out = build_single(src, materials_ref="testref")
    assert out == tmp_path / "tools" / "colab_timing.ipynb"
    nb = nbformat.read(out, as_version=4)
    assert "testref" in nb.cells[0].source
    assert "LogisticRegression().fit" in out.read_text()
    assert not (tmp_path / "reference").exists()
    with pytest.raises(FileNotFoundError):
        build_single(tmp_path / "missing.py")


def test_committed_starters_have_no_solutions():
    for p in (task_dir() / "notebooks").glob("*.ipynb"):
        text = p.read_text()
        assert "SOLUTION" not in text and "### STARTER" not in text, p.name
