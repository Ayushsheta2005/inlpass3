"""src/task1/lstm_train.py – entry point for task1_lstm subcommand."""

import yaml
from src.task1._common import run


def main(config_path: str, mode: str = "evaluate"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = "lstm"  # enforce model type regardless of yaml value
    run(cfg, mode)
