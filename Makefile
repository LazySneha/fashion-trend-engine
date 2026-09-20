.PHONY: ingest transform analyze charts all test clean

VENV := .venv/bin

ingest:
	$(VENV)/python -m src.ingest_gdelt

transform:
	$(VENV)/python -m src.transform

analyze:
	$(VENV)/python -m src.analyze

charts:
	$(VENV)/python -m src.charts

all: ingest transform analyze charts

test:
	$(VENV)/python -m pytest -q

clean:
	rm -rf data/staging data/marts data/output charts
