"""
WandB (Weights & Biases) initialisation and logging helpers.
"""

import os
from typing import Any, Dict, Optional


def init_wandb(
    project: str,
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = None,
) -> Any:
    """
    Initialise a WandB run.

    Args:
        project:  WandB project name
        run_name: human-readable run name
        config:   hyperparameter dict to log
        entity:   WandB entity (team/user). If None, uses default.

    Returns:
        wandb run object, or None if wandb is not installed.
    """
    try:
        import wandb

        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            entity=entity,
        )
        print(f"[WandB] Run initialised: {wandb.run.url}")
        return run
    except ImportError:
        print("[WandB] wandb not installed – logging disabled.")
        return None
    except Exception as e:
        print(f"[WandB] Init failed: {e} – continuing without logging.")
        return None


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log a dict of metrics to WandB (silently skips if not initialised)."""
    try:
        import wandb

        if wandb.run is not None:
            if step is not None:
                metrics = dict(metrics, step=step)
            wandb.log(metrics)
    except Exception:
        pass


def finish_wandb() -> None:
    """Finish the current WandB run."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
