#!/usr/bin/env bash
set -e
python -m src.ingest.bootstrap_data
python -m src.run_pipeline
echo "Signals ready. Launch API:  uvicorn src.api.main:app --reload"
echo "Launch dashboard:         streamlit run src/dashboard/app.py"
