VENV := .venv
PY := $(VENV)/bin/python

.PHONY: env test data dataset notebooks reference verify-reference

env:
	uv venv --allow-existing --python 3.12 $(VENV)
	uv pip install --python $(PY) -r requirements.txt

test:
	$(PY) -m pytest -m "not network"

data:
	$(PY) -m scripts.download_data --out data

dataset:
	$(PY) -m scripts.build_dataset

# Build full notebooks (reference/build) and public starters (notebooks) from reference/notebooks/*.py.
notebooks:
	$(PY) -m scripts.build_notebooks

# Execute reference/build notebooks in numeric order and stop on the first failure.
# NB="02 03" runs only those number prefixes. Wall time per notebook goes to reference/executed/timings.tsv.
reference: notebooks
	@mkdir -p reference/executed reference/outputs
	@test -f reference/executed/timings.tsv || printf 'notebook\tseconds\tfinished_at\n' > reference/executed/timings.tsv
	@set -e; \
	files=$$(ls reference/build/*.ipynb 2>/dev/null | sort); \
	test -n "$$files" || { echo "No notebooks in reference/build"; exit 1; }; \
	for f in $$files; do \
	  n=$$(basename $$f .ipynb); \
	  if [ -n "$(NB)" ]; then case " $(NB) " in *" $${n%%_*} "*) ;; *) continue;; esac; fi; \
	  echo "Running $$n"; start=$$(date +%s); \
	  ENZYME_WORK_DIR="$(CURDIR)/reference/outputs" $(VENV)/bin/jupyter nbconvert --to notebook --execute \
	    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
	    --output-dir reference/executed $$f; \
	  end=$$(date +%s); \
	  printf '%s\t%s\t%s\n' "$$n" "$$((end - start))" "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> reference/executed/timings.tsv; \
	done

# Fail if any executed notebook still shows an unfinished check ("Not yet:") or none exist.
verify-reference:
	@set -e; \
	files=$$(ls reference/executed/*.ipynb 2>/dev/null || true); \
	test -n "$$files" || { echo "No executed notebooks in reference/executed"; exit 1; }; \
	bad=$$(grep -l 'Not yet:' $$files || true); \
	if [ -n "$$bad" ]; then echo "Unfinished checks (Not yet:) in:"; echo "$$bad"; exit 1; fi; \
	echo "All executed notebooks clean."
