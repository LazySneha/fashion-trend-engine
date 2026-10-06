.PHONY: ingest transform analyze charts all test clean agent eval eval-dry

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

# Ask the marts a question. Needs ANTHROPIC_API_KEY; never calls GDELT.
#   make agent Q="which trends reached the affordable tier fastest?"
agent:
	$(VENV)/python -m src.agent --show-sql $(Q)

# Eval suite. `eval-dry` prints the cases and their ground truth with no API calls.
eval:
	$(VENV)/python -m src.agent_eval

eval-dry:
	$(VENV)/python -m src.agent_eval --dry-run

test:
	$(VENV)/python -m pytest -q

clean:
	rm -rf data/staging data/marts data/output charts
