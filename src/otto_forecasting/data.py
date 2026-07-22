from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_TYPES = ("clicks", "carts", "orders")
FEATURE_COLUMNS = (
    "log_clicks",
    "log_carts",
    "log_orders",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
)
TARGET_COLUMNS = ("carts", "orders")


def aggregate_jsonl(input_path: str | Path, frequency: str = "1h") -> pd.DataFrame:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    interval_ms = int(pd.Timedelta(frequency).total_seconds() * 1000)
    counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {event_type: 0 for event_type in EVENT_TYPES}
    )

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                session = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            for event in session.get("events", []):
                event_type = event.get("type")
                timestamp = event.get("ts")
                if event_type not in EVENT_TYPES or timestamp is None:
                    continue
                bucket = int(timestamp) // interval_ms * interval_ms
                counts[bucket][event_type] += 1

    if not counts:
        raise ValueError("No supported OTTO events were found")

    rows = [
        {"timestamp": timestamp, **event_counts}
        for timestamp, event_counts in counts.items()
    ]
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    full_range = pd.date_range(
        start=frame["timestamp"].min(),
        end=frame["timestamp"].max(),
        freq=frequency,
        tz="UTC",
    )
    frame = (
        frame.set_index("timestamp")
        .reindex(full_range, fill_value=0)
        .rename_axis("timestamp")
        .reset_index()
    )
    return frame


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", *EVENT_TYPES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    hour = result["timestamp"].dt.hour.to_numpy()
    day = result["timestamp"].dt.dayofweek.to_numpy()
    result["log_clicks"] = np.log1p(result["clicks"].to_numpy(dtype=np.float64))
    result["log_carts"] = np.log1p(result["carts"].to_numpy(dtype=np.float64))
    result["log_orders"] = np.log1p(result["orders"].to_numpy(dtype=np.float64))
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_sin"] = np.sin(2 * np.pi * day / 7)
    result["day_cos"] = np.cos(2 * np.pi * day / 7)
    result["is_weekend"] = (day >= 5).astype(np.float32)
    return result


def save_processed(frame: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def load_processed(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Processed dataset not found: {source}")
    frame = pd.read_csv(source)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame
