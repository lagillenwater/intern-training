import pytest

from helpers.session import in_colab, require_file, setup_workdir


def test_setup_workdir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ENZYME_WORK_DIR", str(tmp_path))
    p = setup_workdir("run1")
    assert p == tmp_path / "run1" and (p / "figures").is_dir()


def test_setup_workdir_local(tmp_path, monkeypatch):
    monkeypatch.delenv("ENZYME_WORK_DIR", raising=False)
    monkeypatch.setattr("helpers.session.task_dir", lambda: tmp_path)
    p = setup_workdir("run2")
    assert p == tmp_path / "outputs" / "run2" and (p / "figures").is_dir()


def test_in_colab_false_locally():
    assert in_colab() is False


def test_require_file_message(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        require_file(tmp_path / "predictions_baseline.tsv", made_by="Notebook 02 (baseline)")
    msg = str(e.value)
    assert "predictions_baseline.tsv" in msg and "Notebook 02" in msg and "catch-up" in msg
    assert str(tmp_path) not in msg


def test_require_file_returns_existing(tmp_path):
    f = tmp_path / "a.tsv"
    f.write_text("x")
    assert require_file(f, made_by="Notebook 01") == f
