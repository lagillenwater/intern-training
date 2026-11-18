VENV := .venv
PY := $(VENV)/bin/python

.PHONY: env test data dataset

env:
	uv venv --allow-existing --python 3.12 $(VENV)
	uv pip install --python $(PY) -r requirements.txt

test:
	$(PY) -m pytest -m "not network"

data:
	$(PY) -m scripts.download_data --out data

dataset:
	$(PY) -m scripts.build_dataset
