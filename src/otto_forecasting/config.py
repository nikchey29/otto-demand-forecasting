from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    processed_path: str
    frequency: str
    lookback: int
    horizon: int
    validation_steps: int
    test_steps: int
    seasonality: int


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    gradient_clip: float
    num_workers: int


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    output_dir: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProjectConfig(
        seed=int(raw["seed"]),
        output_dir=str(raw["output_dir"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
