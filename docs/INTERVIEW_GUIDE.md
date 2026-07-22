# Interview Guide

## Thirty-second explanation

I built a demand-forecasting system on the OTTO e-commerce clickstream dataset. The pipeline streams the raw JSONL data, aggregates event volume hourly, fills missing intervals, creates cyclical time features, and builds leakage-safe rolling windows. A compact PyTorch Transformer predicts carts and orders for the next 24 hours. I evaluate it in original count units against persistence, seasonal-naive, and Ridge baselines, then save reproducible artifacts and serve the trained model with FastAPI.

## Why a Transformer

The model can learn interactions across a full week of hourly history without recurrent processing. I deliberately use a small hidden dimension and pooled output head because hourly aggregation produces a limited number of calendar observations. The model is valuable only when it beats simpler baselines, so the repository treats baseline comparison as a required experiment.

## How leakage is prevented

The project splits by target period. A training forecast must finish before validation begins, and a validation forecast must finish before testing begins. Historical inputs may come from earlier periods because that information would have been available at forecast time. Both scalers are fitted only on observations before validation.

## Why log-transform counts

Clickstream counts are non-negative and can be highly skewed. `log1p` reduces the influence of extreme peaks and handles zeros. Predictions are inverse-transformed before reporting business-readable metrics.

## Why WAPE

MAE gives an absolute error in event counts. WAPE expresses total absolute error relative to total observed volume, making it easier to compare carts and orders without the instability that ordinary percentage error has near zero.

## What I would improve next

- Forecast top product groups instead of only global volume
- Add uncertainty intervals with quantile loss
- Use rolling-origin backtesting across multiple cutoffs
- Add external features such as promotion and inventory signals
- Track experiments with MLflow
- Add drift monitoring and scheduled retraining

## Questions to expect

### Why not random train-test splitting?

Random splitting would allow later traffic patterns to influence training and would overstate real forecasting performance.

### Why can validation inputs contain training-period observations?

At the moment a validation forecast is issued, earlier historical observations are available. The important restriction is that validation targets are not used for training or preprocessing.

### What happens if Ridge beats the Transformer?

That is a valid experimental result. I would present Ridge as the best model and explain that the data volume did not justify the neural model. Honest model selection is stronger than forcing a Transformer into production.

### How would you productionize it?

I would schedule aggregation, validate schema and data freshness, load a versioned artifact bundle, expose forecasts through the API, monitor error and drift, and retrain only after a controlled backtest and deployment gate.
