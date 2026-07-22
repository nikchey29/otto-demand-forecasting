from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from otto_forecasting.baselines import (
    RidgeForecaster,
    persistence_forecast,
    seasonal_forecast,
)
from otto_forecasting.config import ProjectConfig
from otto_forecasting.data import FEATURE_COLUMNS, TARGET_COLUMNS, add_time_features, load_processed
from otto_forecasting.dataset import build_loaders, inverse_targets, prepare_arrays
from otto_forecasting.metrics import horizon_mae, regression_metrics
from otto_forecasting.model import DemandTransformer
from otto_forecasting.reporting import (
    plot_forecast_example,
    plot_horizon_error,
    plot_model_comparison,
    plot_training_history,
)
from otto_forecasting.training import predict_model, save_history, set_seed, train_model


def run_training(config: ProjectConfig) -> dict[str, Path]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = add_time_features(load_processed(config.data.processed_path))
    feature_values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    target_values = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    prepared = prepare_arrays(
        feature_values=feature_values,
        raw_target_values=target_values,
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        validation_steps=config.data.validation_steps,
        test_steps=config.data.test_steps,
    )
    train_loader, validation_loader, test_loader = build_loaders(
        prepared=prepared,
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandTransformer(
        num_features=len(FEATURE_COLUMNS),
        target_dim=len(TARGET_COLUMNS),
        horizon=config.data.horizon,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
    ).to(device)
    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        patience=config.training.patience,
        gradient_clip=config.training.gradient_clip,
        device=device,
    )
    scaled_prediction, scaled_actual = predict_model(model, test_loader, device)
    transformer_prediction = inverse_targets(scaled_prediction, prepared.target_scaler)
    transformer_actual = inverse_targets(scaled_actual, prepared.target_scaler)

    ridge = RidgeForecaster(alpha=10.0).fit(
        prepared.features,
        prepared.targets,
        prepared.train_starts,
        config.data.lookback,
        config.data.horizon,
    )
    ridge_prediction = inverse_targets(
        ridge.predict(
            prepared.features,
            prepared.test_starts,
            config.data.lookback,
        ),
        prepared.target_scaler,
    )
    persistence_prediction = persistence_forecast(
        prepared.raw_targets,
        prepared.test_starts,
        config.data.horizon,
    )
    seasonal_prediction = seasonal_forecast(
        prepared.raw_targets,
        prepared.test_starts,
        config.data.horizon,
        config.data.seasonality,
    )

    evaluations = {
        "Transformer": transformer_prediction,
        "Ridge": ridge_prediction,
        "Seasonal naive": seasonal_prediction,
        "Persistence": persistence_prediction,
    }
    metric_frames: list[pd.DataFrame] = []
    for name, prediction in evaluations.items():
        metrics = regression_metrics(transformer_actual, prediction, TARGET_COLUMNS)
        metrics.insert(0, "model", name)
        metric_frames.append(metrics)
    comparison = pd.concat(metric_frames, ignore_index=True)
    horizon = horizon_mae(transformer_actual, transformer_prediction, TARGET_COLUMNS)

    model_path = output_dir / "transformer.pt"
    torch.save(model.state_dict(), model_path)
    joblib.dump(prepared.feature_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(prepared.target_scaler, output_dir / "target_scaler.joblib")
    joblib.dump(ridge, output_dir / "ridge.joblib")
    save_history(history, output_dir / "training_history.json")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    horizon.to_csv(output_dir / "horizon_metrics.csv", index=False)

    metadata = {
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "lookback": config.data.lookback,
        "horizon": config.data.horizon,
        "model": config.to_dict()["model"],
        "test_windows": int(len(prepared.test_starts)),
        "device": str(device),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    predictions = []
    for window_index, target_start in enumerate(prepared.test_starts):
        for horizon_index in range(config.data.horizon):
            timestamp_index = target_start + horizon_index
            for target_index, target in enumerate(TARGET_COLUMNS):
                predictions.append(
                    {
                        "window": window_index,
                        "forecast_hour": horizon_index + 1,
                        "timestamp": frame.iloc[timestamp_index]["timestamp"],
                        "target": target,
                        "actual": transformer_actual[window_index, horizon_index, target_index],
                        "prediction": transformer_prediction[
                            window_index, horizon_index, target_index
                        ],
                    }
                )
    pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False)

    plot_training_history(history, output_dir / "training_history.png")
    plot_model_comparison(comparison, output_dir / "model_comparison.png")
    plot_horizon_error(horizon, output_dir / "horizon_mae.png")
    for target_index, target in enumerate(TARGET_COLUMNS):
        plot_forecast_example(
            transformer_actual,
            transformer_prediction,
            target,
            target_index,
            output_dir / f"forecast_{target}.png",
        )

    return {
        "model": model_path,
        "metrics": output_dir / "model_comparison.csv",
        "predictions": output_dir / "predictions.csv",
        "metadata": output_dir / "metadata.json",
    }
