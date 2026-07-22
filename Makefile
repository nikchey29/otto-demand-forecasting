install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

aggregate:
	python -m otto_forecasting.cli aggregate --input data/raw/otto-recsys-train.jsonl --output data/processed/otto_hourly.csv --frequency 1h

train:
	python -m otto_forecasting.cli train --config configs/default.yaml

run:
	python -m otto_forecasting.cli run-all --config configs/default.yaml

api:
	uvicorn otto_forecasting.api:app --host 0.0.0.0 --port 8000
