import numpy as np

from otto_forecasting.metrics import horizon_mae, regression_metrics


def test_metrics_are_zero_for_perfect_prediction():
    actual = np.ones((3, 24, 2))
    metrics = regression_metrics(actual, actual.copy(), ("carts", "orders"))
    assert metrics[["mae", "rmse", "wape", "smape", "mape"]].to_numpy().sum() == 0
    horizon = horizon_mae(actual, actual.copy(), ("carts", "orders"))
    assert horizon["mae"].sum() == 0
