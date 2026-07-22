from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: list[dict[str, float | int]], output: str | Path) -> None:
    frame = pd.DataFrame(history)
    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    axis.plot(frame["epoch"], frame["train_loss"], label="Training")
    axis.plot(frame["epoch"], frame["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Huber loss")
    axis.set_title("Training history")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_model_comparison(metrics: pd.DataFrame, output: str | Path) -> None:
    summary = metrics.groupby("model", as_index=False)["mae"].mean()
    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    axis.bar(summary["model"], summary["mae"])
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean absolute error")
    axis.set_title("Average test MAE across targets")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_horizon_error(horizon_metrics: pd.DataFrame, output: str | Path) -> None:
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    for target, group in horizon_metrics.groupby("target"):
        axis.plot(group["forecast_hour"], group["mae"], label=target)
    axis.set_xlabel("Forecast hour")
    axis.set_ylabel("MAE")
    axis.set_title("Transformer error by forecast horizon")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_forecast_example(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_name: str,
    target_index: int,
    output: str | Path,
) -> None:
    hours = np.arange(1, actual.shape[1] + 1)
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    axis.plot(hours, actual[0, :, target_index], marker="o", label="Actual")
    axis.plot(hours, predicted[0, :, target_index], marker="x", label="Forecast")
    axis.set_xlabel("Forecast hour")
    axis.set_ylabel(target_name.title())
    axis.set_title(f"24-hour {target_name} forecast")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
