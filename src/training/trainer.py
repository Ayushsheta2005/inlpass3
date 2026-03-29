"""
Generic training loop used by all tasks.

Features:
    - Standard train / validation epoch loops
    - Gradient clipping (as in Sutskever et al. 2014)
    - WandB integration via the wandb_utils module
    - Best-model checkpointing
    - Truncated BPTT (detach hidden states every `tbptt_steps`)
"""

import os
import math
import time
from typing import Callable, Dict, Optional
from tqdm import tqdm
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Flexible trainer for sequence models.

    Args:
        model:                   the PyTorch model to train
        optimizer:               pre-constructed optimizer
        criterion:               loss function (e.g. nn.CrossEntropyLoss)
        device:                  torch device
        checkpoint_dir:          directory to save best model checkpoint
        grad_clip:               gradient norm clipping threshold (Sutskever: 5.0)
        use_wandb:               whether to log metrics to WandB
        scheduler:               optional LR scheduler (step called per epoch)
        tbptt_steps:             truncated BPTT window; 0 = no truncation (full BPTT)
        early_stopping_patience: stop training if val loss does not improve for
                                 this many consecutive epochs. 0 disables early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "outputs/logs",
        grad_clip: float = 5.0,
        use_wandb: bool = True,
        scheduler: Optional[object] = None,
        tbptt_steps: int = 0,
        early_stopping_patience: int = 7,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.use_wandb = use_wandb
        self.scheduler = scheduler
        self.tbptt_steps = tbptt_steps
        self.early_stopping_patience = early_stopping_patience

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_loss = math.inf
        self.best_model_path: Optional[str] = None
        self._no_improve_epochs: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clip_and_step(self) -> float:
        """Clip gradients and return the pre-clip gradient norm."""
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        ).item()
        self.optimizer.step()
        return grad_norm

    @staticmethod
    def _detach_state(state):
        """Detach hidden states for truncated BPTT."""
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.detach()
        if isinstance(state, (tuple, list)):
            return type(state)(_Trainer_detach(s) for s in state)
        return state

    # ------------------------------------------------------------------
    # Epoch runners
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        log_interval: int = 50,
        step_fn: Optional[Callable] = None,
    ) -> float:
        """
        Run one training epoch.

        Args:
            loader:       DataLoader yielding batches
            epoch:        current epoch number (for logging)
            log_interval: log every N batches
            step_fn:      optional custom step function(model, batch) → loss.
                          If None, a default cross-entropy step is used.

        Returns:
            mean training loss over the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            if step_fn is not None:
                loss = step_fn(self.model, batch, self.device)
            else:
                loss = self._default_step(batch)

            loss.backward()
            grad_norm = self._clip_and_step()
            total_loss += loss.item()
            n_batches += 1

            if batch_idx % log_interval == 0:
                avg = total_loss / n_batches
                pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.3f}")
                
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log(
                            {
                                "train/batch_loss": loss.item(),
                                "train/grad_norm": grad_norm,
                                "epoch": epoch,
                            }
                        )
                    except Exception:
                        pass

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def val_epoch(
        self,
        loader: DataLoader,
        step_fn: Optional[Callable] = None,
    ) -> float:
        """
        Run one validation epoch.

        Returns:
            mean validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc="[Val]", leave=False)
        for batch in pbar:
            if step_fn is not None:
                loss = step_fn(self.model, batch, self.device)
            else:
                loss = self._default_step(batch)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Default step (seq2seq char-level: batch = (cipher, plain))
    # ------------------------------------------------------------------

    def _default_step(self, batch) -> torch.Tensor:
        cipher, plain = batch
        cipher = cipher.to(self.device)
        plain  = plain.to(self.device)
        logits, _ = self.model(cipher)
        # logits: (B, T, V)  → reshape for cross-entropy
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            plain.reshape(-1),
        )
        return loss

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        model_name: str = "model",
        step_fn: Optional[Callable] = None,
        log_interval: int = 50,
    ) -> Dict[str, list]:
        """
        Train for `n_epochs` with full train/val loop and checkpointing.

        Args:
            train_loader: training data loader
            val_loader:   validation data loader
            n_epochs:     number of epochs
            model_name:   prefix for checkpoint filename
            step_fn:      optional custom step function
            log_interval: batch-level log interval

        Returns:
            history dict with keys 'train_loss', 'val_loss'
        """
        history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(
                train_loader, epoch, log_interval, step_fn
            )
            val_loss = self.val_epoch(val_loader, step_fn)

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log(
                        {
                            "train/epoch_loss": train_loss,
                            "val/epoch_loss": val_loss,
                            "epoch": epoch,
                        }
                    )
                except Exception:
                    pass

            # Save best model / early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._no_improve_epochs = 0
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{model_name}_best.pt"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                self.best_model_path = ckpt_path
                print(f"  ✓ Best model saved → {ckpt_path}")
            else:
                self._no_improve_epochs += 1
                if (
                    self.early_stopping_patience > 0
                    and self._no_improve_epochs >= self.early_stopping_patience
                ):
                    print(
                        f"  ✗ Early stopping triggered — no val improvement "
                        f"for {self._no_improve_epochs} epochs."
                    )
                    break

        return history

    def load_best(self) -> None:
        """Load the best checkpoint into the model in-place."""
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.best_model_path}")


# Standalone detach helper (avoids closure issue with nested method names)
def _Trainer_detach(s):
    if isinstance(s, torch.Tensor):
        return s.detach()
    if isinstance(s, (tuple, list)):
        return type(s)(_Trainer_detach(x) for x in s)
    return s
