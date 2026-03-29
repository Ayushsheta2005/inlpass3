"""src/task2/ssm_train.py – entry point for task2_ssm subcommand."""

import yaml
from src.task2._common import run


def main(config_path: str, mode: str = "evaluate"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = "ssm"
    run(cfg, mode)
