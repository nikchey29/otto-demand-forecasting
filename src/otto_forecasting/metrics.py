from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for index, target in enumerate(target_names):
        observed = actual[..., index].reshape(-1)
        forecast = predicted[..., index].reshape(-1)
        error = forecast - observed
        absolute_error = np.abs(error)
        denominator = np.maximum(np.abs(observed), 1.0)
        rows.append(
            {
                "target": target,
                "mae": float(absolute_error.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "wape": float(absolute_error.sum() / np.maximum(np.abs(observed).sum(), 1.0)),
                "smape": float(
                    np.mean(
                        2
                        * absolute_error
                        / np.maximum(np.abs(observed) + np.abs(forecast), 1.0)
                    )
                ),
                "mape": float(np.mean(absolute_error / denominator)),
            }
        )
    return pd.DataFrame(rows)


def horizon_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for horizon_index in range(actual.shape[1]):
        for target_index, target in enumerate(target_names):
            error = np.abs(
                predicted[:, horizon_index, target_index]
                - actual[:, horizon_index, target_index]
            )
            rows.append(
                {
                    "forecast_hour": horizon_index + 1,
                    "target": target,
                    "mae": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)
