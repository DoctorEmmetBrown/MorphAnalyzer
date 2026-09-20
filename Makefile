.PHONY: install test lint fmt cov inventory clean

VENV ?= .venv
PY   ?= $(VENV)/bin/python

install:
	uv venv $(VENV)
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=morphanalyzer --cov-report=term-missing

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff check . --fix
	$(PY) -m ruff format .

# Regenere l'inventaire iMorph. IMORPH=/chemin/vers/iMorph3.2
inventory:
	$(PY) tools/inventory.py $(IMORPH) \
	  --markdown docs/INVENTAIRE_IMORPH32.md \
	  --json docs/inventaire_imorph32.json

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
