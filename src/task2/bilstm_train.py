"""src/task2/bilstm_train.py – entry point for task2_bilstm subcommand."""

import yaml
from src.task2._common import run


def main(config_path: str, mode: str = "evaluate"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = "bilstm"
    run(cfg, mode)
