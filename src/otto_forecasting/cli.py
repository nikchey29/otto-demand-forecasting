from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_forecasting.config import load_config
from otto_forecasting.data import aggregate_jsonl, save_processed
from otto_forecasting.pipeline import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="otto-forecast")
    subcommands = parser.add_subparsers(dest="command", required=True)

    aggregate = subcommands.add_parser("aggregate")
    aggregate.add_argument("--input", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--frequency", default="1h")

    train = subcommands.add_parser("train")
    train.add_argument("--config", default="configs/default.yaml")

    run_all = subcommands.add_parser("run-all")
    run_all.add_argument("--config", default="configs/default.yaml")

    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "aggregate":
        frame = aggregate_jsonl(arguments.input, arguments.frequency)
        output = save_processed(frame, arguments.output)
        print(json.dumps({"rows": len(frame), "output": str(output)}, indent=2))
        return

    config = load_config(arguments.config)
    if arguments.command == "run-all":
        frame = aggregate_jsonl(config.data.raw_path, config.data.frequency)
        save_processed(frame, config.data.processed_path)
    outputs = run_training(config)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
