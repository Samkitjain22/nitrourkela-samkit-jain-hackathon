.PHONY: setup data run api dash test clean

setup:
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python -m src.ingest.bootstrap_data

run:
	python -m src.run_pipeline

api:
	uvicorn src.api.main:app --reload --port 8000

dash:
	streamlit run src/dashboard/app.py

test:
	pytest -q

clean:
	rm -f data/signals.csv data/signals.json
