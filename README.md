# OTTO E-commerce Demand Forecasting

A production-style machine-learning project that converts the OTTO clickstream dataset into an hourly operational time series and forecasts cart and order volume for the next 24 hours.

## Why this project stands out

- Streams the raw JSONL dataset without loading hundreds of millions of events into memory
- Normalizes timestamps in UTC and fills missing hourly intervals
- Uses cyclical hour and weekday features
- Fits preprocessing only on the training period
- Splits windows by forecast target period to prevent target leakage
- Compares a Transformer against persistence, seasonal-naive, and Ridge baselines
- Reports MAE, RMSE, WAPE, sMAPE, and horizon-level error in original count units
- Saves reproducible artifacts, predictions, metrics, and plots
- Includes a FastAPI inference service, Dockerfile, tests, and continuous integration

## Architecture

```text
OTTO JSONL
    |
    v
Streaming hourly aggregation
    |
    v
UTC reindexing and feature engineering
    |
    v
Chronological train / validation / test windows
    |
    +--------------------+
    |                    |
    v                    v
Transformer          Baselines
    |                    |
    +---------+----------+
              |
              v
Original-scale evaluation and saved artifacts
```

The Transformer projects eight input features into a compact latent space, applies positional encoding and two encoder blocks, mean-pools the sequence, and directly predicts a 24-by-2 forecast matrix.

## Dataset

Download the OTTO Recommender Systems Dataset from Kaggle and place the training file at:

```text
data/raw/otto-recsys-train.jsonl
```

The raw dataset is intentionally excluded from version control.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the complete pipeline

```bash
python -m otto_forecasting.cli run-all --config configs/default.yaml
```

Run the stages separately:

```bash
python -m otto_forecasting.cli aggregate \
  --input data/raw/otto-recsys-train.jsonl \
  --output data/processed/otto_hourly.csv \
  --frequency 1h

python -m otto_forecasting.cli train --config configs/default.yaml
```

## Outputs

Training creates the following files under `artifacts/`:

```text
transformer.pt
feature_scaler.joblib
target_scaler.joblib
ridge.joblib
metadata.json
training_history.json
model_comparison.csv
horizon_metrics.csv
predictions.csv
training_history.png
model_comparison.png
horizon_mae.png
forecast_carts.png
forecast_orders.png
```

`model_comparison.csv` is the primary result table. Resume claims should use the measured improvement over the strongest baseline from this file.

## API

Start the service after training:

```bash
uvicorn otto_forecasting.api:app --host 0.0.0.0 --port 8000
```

The forecast endpoint expects exactly 168 chronological hourly observations:

```json
{
  "history": [
    {
      "timestamp": "2026-01-01T00:00:00Z",
      "clicks": 12000,
      "carts": 900,
      "orders": 250
    }
  ]
}
```

Endpoints:

```text
GET  /health
POST /forecast
```

## Testing

```bash
pytest -q
```

The test suite covers aggregation, missing intervals, feature generation, split boundaries, scaling, model output shape, and evaluation metrics.

## Docker

```bash
docker build -t otto-demand-forecasting .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  otto-demand-forecasting
```

## Experimental design

The final validation and test periods each contain 96 hours. Every sample uses 168 hours of history to forecast the next 24 hours. Input history may precede a split boundary, but forecast targets never cross backward into an earlier split.

The target counts are transformed with `log1p`, standardized using training-only statistics, and converted back to original event counts before reporting metrics.

## Known limitation

The OTTO training timeline covers only a limited number of calendar weeks after hourly aggregation. The project therefore uses a compact model and baseline comparison rather than treating neural-network complexity as evidence of performance. A future extension could forecast demand for top item groups or use a finer aggregation interval to increase the number of operational observations.

## Resume template

Use only the figures produced by your completed run:

> Streamed and aggregated 220M+ e-commerce events into a leakage-aware hourly forecasting pipeline; developed a PyTorch Transformer for 24-hour cart and order prediction and improved test MAE by **[X%]** over the strongest seasonal and linear baseline.

## Repository structure

```text
configs/
data/
src/otto_forecasting/
tests/
.github/workflows/
Dockerfile
Makefile
pyproject.toml
requirements.txt
```
