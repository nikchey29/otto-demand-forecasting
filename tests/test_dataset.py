import numpy as np

from otto_forecasting.dataset import ForecastDataset, make_target_starts, prepare_arrays


def test_make_target_starts_are_separated_by_target_period():
    train, validation, test = make_target_starts(672, 168, 24, 96, 96)
    assert train.max() + 24 <= validation.min()
    assert validation.max() + 24 <= test.min()
    assert test.max() == 648


def test_forecast_dataset_shapes():
    features = np.random.default_rng(1).normal(size=(300, 8)).astype(np.float32)
    targets = np.random.default_rng(2).normal(size=(300, 2)).astype(np.float32)
    dataset = ForecastDataset(features, targets, np.array([168]), 168, 24)
    x, y = dataset[0]
    assert x.shape == (168, 8)
    assert y.shape == (24, 2)


def test_scalers_fit_before_validation_period():
    features = np.arange(800 * 8, dtype=np.float64).reshape(800, 8)
    targets = np.arange(800 * 2, dtype=np.float64).reshape(800, 2)
    prepared = prepare_arrays(features, targets, 168, 24, 96, 96)
    assert len(prepared.train_starts) > 0
    assert prepared.feature_scaler.n_features_in_ == 8
