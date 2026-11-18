VENV := .venv
PY := $(VENV)/bin/python

.PHONY: env test

env:
	uv venv --allow-existing --python 3.12 $(VENV)
	uv pip install --python $(PY) -r requirements.txt

test:
	$(PY) -m pytest -m "not network"
