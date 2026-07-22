# Model Card

## Model

A compact Transformer encoder that uses 168 hourly observations to forecast cart and order volume for the following 24 hours.

## Intended use

- Demonstrating leakage-aware time-series forecasting
- Comparing neural and classical forecasting approaches
- Producing short-horizon operational demand estimates
- Serving forecasts through a small inference API

## Inputs

Each hourly record contains click, cart, and order counts. The pipeline derives log-count, cyclical time, and weekend features.

## Outputs

A 24-step forecast for:

- Cart volume
- Order volume

Predictions are converted back to non-negative event counts before evaluation and serving.

## Evaluation

The project evaluates the Transformer, Ridge regression, seasonal-naive forecasting, and persistence forecasting on the same isolated test windows.

Primary metrics:

- MAE
- RMSE
- WAPE
- sMAPE
- MAPE
- MAE by forecast hour

Measured results are written to `artifacts/model_comparison.csv` and `artifacts/horizon_metrics.csv`.

## Data leakage controls

- Chronological train, validation, and test periods
- Every forecast target window remains fully inside its assigned split
- Feature and target scalers are fitted only on the training period
- Test metrics are computed after model selection is complete

## Limitations

- Hourly aggregation creates a relatively short calendar series
- Events are aggregated globally rather than by product or customer segment
- The model does not use price, promotion, inventory, or marketing data
- Forecast uncertainty intervals are not currently estimated
- Performance on future production traffic is not established by the public dataset alone

## Responsible use

The model is an educational and portfolio system. It should not be used for production inventory or staffing decisions without monitoring, retraining, uncertainty estimation, and validation on current business data.
