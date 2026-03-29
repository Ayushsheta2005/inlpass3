"""
Task 1: Cipher Decryption
==========================
Train an RNN and LSTM from scratch to map cipher text → plain text.

Usage:
    python -m src.task1_train --model rnn  [options]
    python -m src.task1_train --model lstm [options]

Outputs:
    outputs/results/task1_rnn.txt   – decoded output from RNN
    outputs/results/task1_lstm.txt  – decoded output from LSTM
    outputs/logs/<model>_best.pt    – best checkpoint

WandB run and HuggingFace push are performed on completion.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running as `python -m src.task1_train` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import (
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
from src.utils.wandb_utils import finish_wandb, init_wandb, log_metrics
from src.utils.huggingface import push_model, save_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Task 1 – Cipher Decryption")
    p.add_argument("--model", choices=["rnn", "lstm"], required=True)
    p.add_argument("--plain",  default="data/plain.txt")
    p.add_argument("--cipher", default="data/cipher_00.txt")
    p.add_argument("--seq_len",     type=int, default=256)
    p.add_argument("--embed_dim",   type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers",  type=int, default=2)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--grad_clip",   type=float, default=5.0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--wandb_project", default="INLP_A3")
    p.add_argument("--hf_repo",     default=None,
                   help="HuggingFace repo ID for model push, e.g. user/model")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--output_dir",  default="outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------

def make_step_fn(pad_idx: int):
    """Returns a batch step function compatible with Trainer."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def step(model, batch, device):
        cipher, plain = batch
        cipher = cipher.to(device)
        plain  = plain.to(device)
        logits, _ = model(cipher)           # (B, T, V)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            plain.reshape(-1),
        )
        return loss

    return step


# ---------------------------------------------------------------------------
# Decoding / greedy inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_dataset(model, loader, idx2char, device):
    """Greedy character-level decoding over the full dataset."""
    model.eval()
    all_preds, all_targets = [], []

    for cipher, plain in loader:
        cipher = cipher.to(device)
        logits, _ = model(cipher)           # (B, T, V)
        pred_ids = logits.argmax(dim=-1)    # (B, T)

        for b in range(pred_ids.size(0)):
            pred_str = decode_chars(pred_ids[b].tolist(), idx2char)
            tgt_str  = decode_chars(plain[b].tolist(), idx2char)
            all_preds.append(pred_str)
            all_targets.append(tgt_str)

    return all_preds, all_targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(f"{args.output_dir}/results", exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs",    exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading and preprocessing
    # ------------------------------------------------------------------
    plain_text  = clean_text(load_text(args.plain))
    cipher_text = clean_text(load_text(args.cipher))

    # Build vocab from both texts to cover all characters
    char2idx, idx2char = build_char_vocab([plain_text, cipher_text])
    vocab_size = len(char2idx)
    pad_idx    = char2idx["<PAD>"]
    print(f"Vocab size: {vocab_size}")

    samples = build_parallel_samples(
        cipher_text, plain_text, char2idx, seq_len=args.seq_len
    )
    print(f"Total samples: {len(samples)}")

    train_s, val_s, test_s = split_data(samples, 0.8, 0.1)
    print(f"Train={len(train_s)}  Val={len(val_s)}  Test={len(test_s)}")

    train_ds = CipherDataset(train_s)
    val_ds   = CipherDataset(val_s)
    test_ds  = CipherDataset(test_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = dict(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=pad_idx,
    )

    if args.model == "rnn":
        model = RNN(**model_kwargs)
    else:
        model = LSTM(**model_kwargs)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}  |  Params: {n_params:,}")

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        init_wandb(
            project=args.wandb_project,
            run_name=f"task1_{args.model}",
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(ignore_index=pad_idx),
        device=device,
        checkpoint_dir=f"{args.output_dir}/logs",
        grad_clip=args.grad_clip,
        use_wandb=use_wandb,
        scheduler=scheduler,
    )

    step_fn = make_step_fn(pad_idx)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        model_name=f"task1_{args.model}",
        step_fn=step_fn,
        log_interval=20,
    )

    # Load best checkpoint for evaluation
    trainer.load_best()

    # ------------------------------------------------------------------
    # Evaluation on test split
    # ------------------------------------------------------------------
    preds, targets = decode_dataset(model, test_loader, idx2char, device)
    metrics = compute_task1_metrics(preds, targets)

    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if use_wandb:
        log_metrics({f"test/{k}": v for k, v in metrics.items()})

    # ------------------------------------------------------------------
    # Save output text
    # ------------------------------------------------------------------
    result_path = f"{args.output_dir}/results/task1_{args.model}.txt"
    with open(result_path, "w") as f:
        for pred, tgt in zip(preds, targets):
            f.write(f"[CIPHER-IN] {tgt[:80]}\n")
            f.write(f"[MODEL-OUT] {pred[:80]}\n")
            f.write("-" * 80 + "\n")
    print(f"Results saved → {result_path}")

    # ------------------------------------------------------------------
    # Save vocab metadata and checkpoint
    # ------------------------------------------------------------------
    meta = {
        "char2idx": char2idx,
        "idx2char": {str(k): v for k, v in idx2char.items()},
        "model_type": args.model,
        "model_config": model_kwargs,
        "metrics": metrics,
    }

    local_ckpt = f"{args.output_dir}/logs/task1_{args.model}_final.pt"
    save_checkpoint(model, meta, local_ckpt)

    # Save meta separately for easy reload
    meta_path = f"{args.output_dir}/logs/task1_{args.model}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # ------------------------------------------------------------------
    # HuggingFace push
    # ------------------------------------------------------------------
    if args.hf_repo:
        push_model(model, meta, repo_id=args.hf_repo)

    finish_wandb()
    print("Done.")


if __name__ == "__main__":
    main()
