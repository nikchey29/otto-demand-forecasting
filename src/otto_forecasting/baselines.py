from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class RidgeForecaster:
    def __init__(self, alpha: float = 10.0) -> None:
        self.model = Ridge(alpha=alpha)
        self.horizon = 0
        self.target_dim = 0

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> "RidgeForecaster":
        x = np.stack(
            [features[start - lookback : start].reshape(-1) for start in target_starts]
        )
        y = np.stack(
            [targets[start : start + horizon].reshape(-1) for start in target_starts]
        )
        self.horizon = horizon
        self.target_dim = targets.shape[1]
        self.model.fit(x, y)
        return self

    def predict(
        self,
        features: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
    ) -> np.ndarray:
        x = np.stack(
            [features[start - lookback : start].reshape(-1) for start in target_starts]
        )
        prediction = self.model.predict(x)
        return prediction.reshape(-1, self.horizon, self.target_dim)


def persistence_forecast(
    raw_targets: np.ndarray,
    target_starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    return np.stack(
        [np.repeat(raw_targets[start - 1][None, :], horizon, axis=0) for start in target_starts]
    )


def seasonal_forecast(
    raw_targets: np.ndarray,
    target_starts: np.ndarray,
    horizon: int,
    seasonality: int,
) -> np.ndarray:
    if seasonality < horizon:
        raise ValueError("Seasonality must be at least as large as the forecast horizon")
    return np.stack(
        [raw_targets[start - seasonality : start - seasonality + horizon] for start in target_starts]
    )
