"""src/task1/rnn_train.py – entry point for task1_rnn subcommand."""

import yaml
from src.task1._common import run


def main(config_path: str, mode: str = "evaluate"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = "rnn"   # enforce model type regardless of yaml value
    run(cfg, mode)
