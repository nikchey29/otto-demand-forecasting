from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class ForecastDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_starts: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.target_starts = target_starts.astype(np.int64, copy=False)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_start = int(self.target_starts[index])
        x = self.features[target_start - self.lookback : target_start]
        y = self.targets[target_start : target_start + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass(frozen=True)
class PreparedData:
    features: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    train_starts: np.ndarray
    validation_starts: np.ndarray
    test_starts: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def make_target_starts(
    length: int,
    lookback: int,
    horizon: int,
    validation_steps: int,
    test_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation_start = length - validation_steps - test_steps
    test_start = length - test_steps
    final_start = length - horizon
    if validation_start <= lookback:
        raise ValueError("Not enough observations for the requested lookback and splits")
    if validation_steps < horizon or test_steps < horizon:
        raise ValueError("Validation and test periods must each be at least one horizon")

    all_starts = np.arange(lookback, final_start + 1, dtype=np.int64)
    train = all_starts[all_starts + horizon <= validation_start]
    validation = all_starts[
        (all_starts >= validation_start) & (all_starts + horizon <= test_start)
    ]
    test = all_starts[all_starts >= test_start]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("One or more dataset splits are empty")
    return train, validation, test


def prepare_arrays(
    feature_values: np.ndarray,
    raw_target_values: np.ndarray,
    lookback: int,
    horizon: int,
    validation_steps: int,
    test_steps: int,
) -> PreparedData:
    train_starts, validation_starts, test_starts = make_target_starts(
        length=len(feature_values),
        lookback=lookback,
        horizon=horizon,
        validation_steps=validation_steps,
        test_steps=test_steps,
    )
    scaler_fit_end = int(validation_starts[0])
    feature_scaler = StandardScaler().fit(feature_values[:scaler_fit_end])
    log_targets = np.log1p(raw_target_values.astype(np.float64))
    target_scaler = StandardScaler().fit(log_targets[:scaler_fit_end])
    features = feature_scaler.transform(feature_values)
    targets = target_scaler.transform(log_targets)
    return PreparedData(
        features=features,
        targets=targets,
        raw_targets=raw_target_values.astype(np.float64),
        train_starts=train_starts,
        validation_starts=validation_starts,
        test_starts=test_starts,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


def build_loaders(
    prepared: PreparedData,
    lookback: int,
    horizon: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = ForecastDataset(
        prepared.features,
        prepared.targets,
        prepared.train_starts,
        lookback,
        horizon,
    )
    validation_dataset = ForecastDataset(
        prepared.features,
        prepared.targets,
        prepared.validation_starts,
        lookback,
        horizon,
    )
    test_dataset = ForecastDataset(
        prepared.features,
        prepared.targets,
        prepared.test_starts,
        lookback,
        horizon,
    )
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    return train_loader, validation_loader, test_loader


def inverse_targets(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = values.shape
    restored = scaler.inverse_transform(values.reshape(-1, shape[-1]))
    return np.maximum(np.expm1(restored), 0.0).reshape(shape)
