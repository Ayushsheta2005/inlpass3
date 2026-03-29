"""
Task 2: Language Modeling
==========================
2a. SSM (S4) for Next Word Prediction (NWP)
2b. Bidirectional LSTM for Masked Language Modeling (MLM)

Usage:
    python -m src.task2_train --model ssm   [options]
    python -m src.task2_train --model bilstm [options]

Outputs:
    outputs/results/task2_ssm.txt    – NWP samples + perplexity
    outputs/results/task2_bilstm.txt – MLM predictions + perplexity
    outputs/logs/<model>_best.pt     – best checkpoint
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import (
    build_lm_samples,
    build_mlm_samples,
    build_word_vocab,
    clean_text,
    decode_words,
    load_text,
    split_data,
)
from src.data.dataset import MLMDataset, NWPDataset
from src.models.ssm import S4Model
from src.models.bilstm import BiLSTM
from src.training.trainer import Trainer
from src.training.metrics import perplexity
from src.utils.wandb_utils import finish_wandb, init_wandb, log_metrics
from src.utils.huggingface import push_model, save_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Task 2 – Language Modeling")
    p.add_argument("--model", choices=["ssm", "bilstm"], required=True)
    p.add_argument("--plain", default="data/plain.txt")
    p.add_argument("--seq_len",     type=int, default=64)
    p.add_argument("--embed_dim",   type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers",  type=int, default=4)
    p.add_argument("--N",           type=int, default=64,
                   help="SSM state dimension (S4 only)")
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--mask_prob",   type=float, default=0.15,
                   help="Masking probability for Bi-LSTM MLM")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--grad_clip",   type=float, default=5.0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--wandb_project", default="INLP_A3")
    p.add_argument("--hf_repo",     default=None)
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--output_dir",  default="outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def nwp_step_fn(pad_idx: int):
    """NWP step: predict next word at every time-step."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def step(model, batch, device):
        inp, tgt = batch
        inp = inp.to(device)
        tgt = tgt.to(device)
        logits = model(inp)             # (B, T, V)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )
        return loss

    return step


def mlm_step_fn(pad_idx: int):
    """MLM step: predict original tokens at masked positions only."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def step(model, batch, device):
        masked_inp, original, mask_flags = batch
        masked_inp  = masked_inp.to(device)
        original    = original.to(device)
        mask_flags  = mask_flags.to(device)

        logits = model(masked_inp)      # (B, T, V)

        # Compute loss only at masked positions
        # Set non-masked positions to pad_idx so they are ignored
        labels = original.clone()
        labels[mask_flags == 0] = pad_idx

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return loss

    return step


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nwp_perplexity(model, loader, pad_idx, device):
    """Compute NWP perplexity on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        logits = model(inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        n_toks = (tgt != pad_idx).sum().item()
        total_loss   += loss.item()
        total_tokens += n_toks

    mean_nll = total_loss / max(total_tokens, 1)
    return perplexity(mean_nll)


@torch.no_grad()
def evaluate_mlm_perplexity(model, loader, pad_idx, device):
    """Compute MLM perplexity over masked positions only."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for masked_inp, original, mask_flags in loader:
        masked_inp = masked_inp.to(device)
        original   = original.to(device)
        mask_flags = mask_flags.to(device)

        logits = model(masked_inp)

        labels = original.clone()
        labels[mask_flags == 0] = pad_idx

        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        n_toks = (labels != pad_idx).sum().item()
        total_loss   += loss.item()
        total_tokens += n_toks

    mean_nll = total_loss / max(total_tokens, 1)
    return perplexity(mean_nll)


# ---------------------------------------------------------------------------
# Sample generation (NWP greedy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_nwp_sample(model, seed_ids, idx2word, max_steps, device):
    """Greedy next-word generation from a seed sequence."""
    model.eval()
    generated = list(seed_ids)
    x = torch.tensor([seed_ids], dtype=torch.long, device=device)

    for _ in range(max_steps):
        logits = model(x)               # (1, T, V)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        x = torch.tensor([generated[-64:]], dtype=torch.long, device=device)
        if idx2word.get(next_id, "") == "<EOS>":
            break

    return decode_words(generated, idx2word)


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
    # Data
    # ------------------------------------------------------------------
    plain_text = clean_text(load_text(args.plain))
    min_freq = getattr(args, "min_freq", 1)
    word2idx, idx2word = build_word_vocab([plain_text], min_freq=min_freq)
    vocab_size = len(word2idx)
    pad_idx    = word2idx["<PAD>"]
    print(f"Word vocab size: {vocab_size} (min_freq={min_freq})")
    print(f"Word vocab size: {vocab_size}")

    if args.model == "ssm":
        samples = build_lm_samples(plain_text, word2idx, seq_len=args.seq_len)
        train_s, val_s, test_s = split_data(samples)
        train_ds = NWPDataset(train_s)
        val_ds   = NWPDataset(val_s)
        test_ds  = NWPDataset(test_s)
    else:
        samples = build_mlm_samples(
            plain_text, word2idx, seq_len=args.seq_len, mask_prob=args.mask_prob
        )
        train_s, val_s, test_s = split_data(samples)
        train_ds = MLMDataset(train_s)
        val_ds   = MLMDataset(val_s)
        test_ds  = MLMDataset(test_s)

    print(f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.model == "ssm":
        model = S4Model(
            vocab_size=vocab_size,
            d_model=args.embed_dim,
            N=args.N,
            n_layers=args.num_layers,
            l_max=args.seq_len,
            dropout=args.dropout,
            pad_idx=pad_idx,
        )
    else:
        model = BiLSTM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=pad_idx,
        )

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
            run_name=f"task2_{args.model}",
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    step_fn = nwp_step_fn(pad_idx) if args.model == "ssm" else mlm_step_fn(pad_idx)

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

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        model_name=f"task2_{args.model}",
        step_fn=step_fn,
        log_interval=20,
    )

    trainer.load_best()

    # ------------------------------------------------------------------
    # Evaluate perplexity
    # ------------------------------------------------------------------
    if args.model == "ssm":
        test_ppl = evaluate_nwp_perplexity(model, test_loader, pad_idx, device)
    else:
        test_ppl = evaluate_mlm_perplexity(model, test_loader, pad_idx, device)

    print(f"\n=== Test Perplexity ({args.model.upper()}): {test_ppl:.2f} ===")

    if use_wandb:
        log_metrics({f"test/perplexity": test_ppl})

    # ------------------------------------------------------------------
    # Write sample output
    # ------------------------------------------------------------------
    result_path = f"{args.output_dir}/results/task2_{args.model}.txt"
    with open(result_path, "w") as f:
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"Task: {'Next Word Prediction' if args.model == 'ssm' else 'Masked Language Modeling'}\n")
        f.write(f"Test Perplexity: {test_ppl:.4f}\n\n")

        if args.model == "ssm":
            # Generate a few sample continuations
            f.write("=== Sample NWP Outputs ===\n")
            words = plain_text.split()[:args.seq_len]
            from src.data.preprocess import encode_words
            seed = encode_words(" ".join(words[:10]), word2idx)
            generated = generate_nwp_sample(model, seed, idx2word, 30, device)
            f.write(f"Seed : {' '.join(words[:10])}\n")
            f.write(f"Generated : {generated}\n")
        else:
            f.write("=== Sample MLM Predictions ===\n")
            for masked_inp, original, mask_flags in test_loader:
                masked_inp = masked_inp.to(device)
                logits = model(masked_inp)
                pred_ids = logits.argmax(dim=-1)
                for b in range(min(3, pred_ids.size(0))):
                    orig_str    = decode_words(original[b].tolist(), idx2word)
                    pred_str    = decode_words(pred_ids[b].tolist(), idx2word)
                    masked_str  = decode_words(masked_inp[b].tolist(), idx2word)
                    f.write(f"Original : {orig_str[:100]}\n")
                    f.write(f"Masked   : {masked_str[:100]}\n")
                    f.write(f"Predicted: {pred_str[:100]}\n")
                    f.write("-" * 80 + "\n")
                break

    print(f"Results saved → {result_path}")

    # ------------------------------------------------------------------
    # Checkpoint + HuggingFace
    # ------------------------------------------------------------------
    vocab_meta = {
        "word2idx": word2idx,
        "idx2word": {str(k): v for k, v in idx2word.items()},
        "model_type": args.model,
        "model_config": {
            "vocab_size": vocab_size,
            "embed_dim": args.embed_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "N": args.N,
            "dropout": args.dropout,
            "seq_len": args.seq_len,
        },
        "test_perplexity": test_ppl,
    }

    save_checkpoint(model, vocab_meta, f"{args.output_dir}/logs/task2_{args.model}_final.pt")
    with open(f"{args.output_dir}/logs/task2_{args.model}_meta.json", "w") as f:
        json.dump(vocab_meta, f)

    if args.hf_repo:
        push_model(model, vocab_meta, repo_id=args.hf_repo)

    finish_wandb()
    print("Done.")


if __name__ == "__main__":
    main()
