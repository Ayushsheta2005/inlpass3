"""
Shared training/evaluation logic for Task 1 (Cipher Decryption).
Both rnn_train.py and lstm_train.py delegate here.
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocess import (
    CIPHER_PAD,
    build_char_vocab,
    build_parallel_samples,
    clean_text,
    decode_chars,
    load_text,
    split_data,
)
from src.data.dataset import CipherDataset
from src.models.rnn import RNN
from src.models.lstm import LSTM
from src.training.trainer import Trainer
from src.training.metrics import compute_task1_metrics
import torch as _torch
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb


def make_step_fn(pad_idx: int):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def step(model, batch, device):
        cipher, plain = batch
        cipher = cipher.to(device)
        plain = plain.to(device)
        logits, _ = model(cipher)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            plain.reshape(-1),
        )
        return loss

    return step


@torch.no_grad()
def decode_dataset(model, loader, idx2char, device):
    from tqdm import tqdm
    model.eval()
    all_preds, all_targets = [], []
    for cipher, plain in tqdm(loader, desc="Decoding Test Set"):
        cipher = cipher.to(device)
        logits, _ = model(cipher)
        pred_ids = logits.argmax(dim=-1)
        for b in range(pred_ids.size(0)):
            tgt_str = decode_chars(plain[b].tolist(), idx2char)
            pred_str = decode_chars(pred_ids[b].tolist(), idx2char)
            # Trim prediction to target length — the model outputs seq_len tokens
            # but padded positions produce garbage; only keep the real content.
            pred_str = pred_str[: len(tgt_str)]
            all_preds.append(pred_str)
            all_targets.append(tgt_str)
    return all_preds, all_targets


def run(cfg: dict, mode: str):
    """Full train/evaluate pipeline for Task 1.

    Args:
        cfg:  dict of config values (loaded from YAML)
        mode: 'train' | 'evaluate' | 'both'
    """
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Task 1 | {cfg['model'].upper()}] device={device}  mode={mode}")

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(f"{out_dir}/results", exist_ok=True)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)

    model_name = f"task1_{cfg['model']}"
    ckpt_best = f"{out_dir}/logs/{model_name}_best.pt"
    meta_path = f"{out_dir}/logs/{model_name}_meta.json"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    plain_text = clean_text(load_text(cfg["plain"]))
    cipher_text = clean_text(load_text(cfg["cipher"]))

    # Build vocabulary from *plain* text only — cipher tokens are integer IDs 0-99.
    char2idx, idx2char = build_char_vocab([plain_text])
    vocab_size = len(char2idx)
    pad_idx = char2idx["<PAD>"]
    print(f"Plain vocab size: {vocab_size}  |  Cipher vocab size: 101 (0-99 + pad=100)")

    samples = build_parallel_samples(
        cipher_text, plain_text, char2idx, seq_len=cfg.get("seq_len", 256)
    )
    train_s, val_s, test_s = split_data(samples, 0.8, 0.1)
    print(f"Train={len(train_s)}  Val={len(val_s)}  Test={len(test_s)}")

    bs = cfg.get("batch_size", 64)
    train_loader = DataLoader(CipherDataset(train_s), batch_size=bs, shuffle=True)
    val_loader = DataLoader(CipherDataset(val_s), batch_size=bs)
    test_loader = DataLoader(CipherDataset(test_s), batch_size=bs)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Cipher tokens: 0-99 (real) + 100 (PAD).  Plain chars: vocab_size entries.
    CIPHER_VOCAB = CIPHER_PAD + 1  # = 101
    model_kwargs = dict(
        vocab_size=vocab_size,
        embed_dim=cfg.get("embed_dim", 128),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
        pad_idx=pad_idx,
        cipher_vocab_size=CIPHER_VOCAB,
        cipher_pad_idx=CIPHER_PAD,
    )
    model = (RNN if cfg["model"] == "rnn" else LSTM)(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    meta = {
        "char2idx": char2idx,
        "idx2char": {str(k): v for k, v in idx2char.items()},
        "model_type": cfg["model"],
        "model_config": model_kwargs,
        "cipher_vocab_size": CIPHER_VOCAB,
        "cipher_pad_idx": CIPHER_PAD,
    }

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = not cfg.get("no_wandb", False)
    if use_wandb and mode in ("train", "both"):
        try:
            init_wandb(project=cfg.get("wandb_project", "INLP_A3"), config=cfg,
                       name=model_name)
        except Exception:
            use_wandb = False

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if mode in ("train", "both"):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg.get("lr_patience", 3),
            factor=cfg.get("lr_factor", 0.5),
        )
        class Task1Trainer(Trainer):
            def val_epoch(self, loader, step_fn=None):
                val_loss = super().val_epoch(loader, step_fn)
                
                # Compute accuracy on validation set
                preds, targets = decode_dataset(self.model, loader, idx2char, self.device)
                metrics = compute_task1_metrics(preds, targets)
                
                print(f"  * [Val Metrics] Char Acc: {metrics['char_acc']:.4f} | Word Acc: {metrics['word_acc']:.4f}")
                
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            "val/char_acc": metrics["char_acc"],
                            "val/word_acc": metrics["word_acc"],
                        }, commit=False)
                    except Exception:
                        pass
                        
                return val_loss

        trainer = Task1Trainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(ignore_index=pad_idx),
            device=device,
            checkpoint_dir=f"{out_dir}/logs",
            grad_clip=cfg.get("grad_clip", 5.0),
            use_wandb=use_wandb,
            scheduler=scheduler,
            early_stopping_patience=cfg.get("early_stopping_patience", 7),
        )
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=cfg.get("epochs", 30),
            model_name=model_name,
            step_fn=make_step_fn(pad_idx),
            log_interval=20,
        )
        trainer.load_best()
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    if mode in ("evaluate", "both"):
        if mode == "evaluate":
            if os.path.isfile(ckpt_best):
                state_dict = torch.load(ckpt_best, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"Loaded checkpoint: {ckpt_best}")
            else:
                print(f"Warning: no checkpoint found at {ckpt_best}, using random weights.")

        preds, targets = decode_dataset(model, test_loader, idx2char, device)
        metrics = compute_task1_metrics(preds, targets)

        print("\n=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if use_wandb:
            try:
                log_wandb({f"test/{k}": v for k, v in metrics.items()})
            except Exception:
                pass

        result_path = f"{out_dir}/results/{model_name}.txt"
        with open(result_path, "w") as f:
            for pred, tgt in zip(preds, targets):
                f.write(f"[CIPHER-IN] {tgt[:80]}\n")
                f.write(f"[MODEL-OUT] {pred[:80]}\n")
                f.write("-" * 80 + "\n")
        print(f"Results saved → {result_path}")
        meta["metrics"] = metrics

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    if use_wandb:
        try:
            finish_wandb()
        except Exception:
            pass

    print("Done.")
