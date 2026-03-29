"""
Shared training/evaluation logic for Task 2 (Language Modeling).
Both ssm_train.py and bilstm_train.py delegate here.
"""

import json
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocess import (
    build_lm_samples,
    build_mlm_samples,
    build_word_vocab,
    clean_text,
    decode_words,
    encode_words,
    load_text,
    split_data,
)
from src.data.dataset import MLMDataset, NWPDataset
from src.models.ssm import S4Model
from src.models.bilstm import BiLSTM
from src.training.trainer import Trainer
from src.training.metrics import perplexity
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def nwp_step_fn(pad_idx: int):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def step(model, batch, device):
        inp, tgt = batch
        inp = inp.to(device)
        tgt = tgt.to(device)
        logits = model(inp)
        return criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

    return step


def mlm_step_fn(pad_idx: int):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def step(model, batch, device):
        masked_inp, original, mask_flags = batch
        masked_inp = masked_inp.to(device)
        original = original.to(device)
        mask_flags = mask_flags.to(device)
        logits = model(masked_inp)
        labels = original.clone()
        labels[mask_flags == 0] = pad_idx
        return criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    return step


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nwp_perplexity(model, loader, pad_idx, device):
    model.eval()
    from tqdm import tqdm
    for inp, tgt in tqdm(loader, desc="Eval NWP", leave=False):
        inp, tgt = inp.to(device), tgt.to(device)
        logits = model(inp)
        total_loss += criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()
        total_tokens += (tgt != pad_idx).sum().item()
    return perplexity(total_loss / max(total_tokens, 1))


@torch.no_grad()
def evaluate_mlm_perplexity(model, loader, pad_idx, device):
    model.eval()
    from tqdm import tqdm
    for masked_inp, original, mask_flags in tqdm(loader, desc="Eval MLM", leave=False):
        masked_inp, original, mask_flags = (
            masked_inp.to(device), original.to(device), mask_flags.to(device)
        )
        logits = model(masked_inp)
        labels = original.clone()
        labels[mask_flags == 0] = pad_idx
        total_loss += criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).item()
        total_tokens += (labels != pad_idx).sum().item()
    return perplexity(total_loss / max(total_tokens, 1))


# ---------------------------------------------------------------------------
# Text generation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_nwp_sample(model, seed_ids, idx2word, max_steps, seq_len, device):
    model.eval()
    generated = list(seed_ids)
    x = torch.tensor([seed_ids], dtype=torch.long, device=device)
    for _ in range(max_steps):
        logits = model(x)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        x = torch.tensor([generated[-seq_len:]], dtype=torch.long, device=device)
        if idx2word.get(next_id, "") == "<EOS>":
            break
    return decode_words(generated, idx2word)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict, mode: str):
    """Full train/evaluate pipeline for Task 2.

    Args:
        cfg:  dict of config values (loaded from YAML)
        mode: 'train' | 'evaluate' | 'both'
    """
    torch.manual_seed(cfg.get("seed", 42))
    _device_cfg = cfg.get("device", None)
    if _device_cfg:
        device = torch.device(_device_cfg)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    model_type = cfg["model"]
    print(f"[Task 2 | {model_type.upper()}] device={device}  mode={mode}")

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(f"{out_dir}/results", exist_ok=True)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)

    model_name = f"task2_{model_type}"
    ckpt_best = f"{out_dir}/logs/{model_name}_best.pt"
    meta_path = f"{out_dir}/logs/{model_name}_meta.json"
    seq_len = cfg.get("seq_len", 64)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    plain_text = clean_text(load_text(cfg["plain"]))
    min_freq = cfg.get("min_freq", 1)
    word2idx, idx2word = build_word_vocab([plain_text], min_freq=min_freq)
    vocab_size = len(word2idx)
    pad_idx = word2idx["<PAD>"]
    print(f"Word vocab size: {vocab_size} (min_freq={min_freq})")

    if model_type == "ssm":
        samples = build_lm_samples(plain_text, word2idx, seq_len=seq_len)
        train_s, val_s, test_s = split_data(samples)
        Ds = NWPDataset
    else:
        samples = build_mlm_samples(
            plain_text, word2idx, seq_len=seq_len,
            mask_prob=cfg.get("mask_prob", 0.15)
        )
        train_s, val_s, test_s = split_data(samples)
        Ds = MLMDataset

    print(f"Train={len(train_s)}  Val={len(val_s)}  Test={len(test_s)}")

    bs = cfg.get("batch_size", 32)
    train_loader = DataLoader(Ds(train_s), batch_size=bs, shuffle=True)
    val_loader = DataLoader(Ds(val_s), batch_size=bs)
    test_loader = DataLoader(Ds(test_s), batch_size=bs)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if model_type == "ssm":
        model = S4Model(
            vocab_size=vocab_size,
            d_model=cfg.get("embed_dim", 256),
            N=cfg.get("N", 64),
            n_layers=cfg.get("num_layers", 4),
            l_max=seq_len,
            dropout=cfg.get("dropout", 0.1),
            pad_idx=pad_idx,
        )
    else:
        model = BiLSTM(
            vocab_size=vocab_size,
            embed_dim=cfg.get("embed_dim", 256),
            hidden_size=cfg.get("hidden_size", 256),
            num_layers=cfg.get("num_layers", 4),
            dropout=cfg.get("dropout", 0.1),
            pad_idx=pad_idx,
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    vocab_meta = {
        "word2idx": word2idx,
        "idx2word": {str(k): v for k, v in idx2word.items()},
        "model_type": model_type,
        "model_config": {
            "vocab_size": vocab_size,
            "embed_dim": cfg.get("embed_dim", 256),
            "hidden_size": cfg.get("hidden_size", 256),
            "num_layers": cfg.get("num_layers", 4),
            "N": cfg.get("N", 64),
            "dropout": cfg.get("dropout", 0.1),
            "seq_len": seq_len,
        },
    }

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = not cfg.get("no_wandb", False)
    if use_wandb and mode in ("train", "both"):
        try:
            init_wandb(project=cfg.get("wandb_project", "INLP_A3"), config=cfg, name=model_name)
        except Exception:
            use_wandb = False

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if mode in ("train", "both"):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg.get("lr_patience", 3),
            factor=cfg.get("lr_factor", 0.5),
        )
        step_fn = nwp_step_fn(pad_idx) if model_type == "ssm" else mlm_step_fn(pad_idx)
        trainer = Trainer(
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
            step_fn=step_fn,
            log_interval=20,
        )
        trainer.load_best()

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
                print(f"Warning: no checkpoint at {ckpt_best}, evaluating with random weights.")

        if model_type == "ssm":
            test_ppl = evaluate_nwp_perplexity(model, test_loader, pad_idx, device)
        else:
            test_ppl = evaluate_mlm_perplexity(model, test_loader, pad_idx, device)

        print(f"\n=== Test Perplexity ({model_type.upper()}): {test_ppl:.2f} ===")

        if use_wandb:
            try:
                log_wandb({"test/perplexity": test_ppl})
            except Exception:
                pass

        # Write result file
        result_path = f"{out_dir}/results/{model_name}.txt"
        with open(result_path, "w") as f:
            f.write(f"Model: {model_type.upper()}\n")
            f.write(f"Task: {'Next Word Prediction' if model_type == 'ssm' else 'Masked Language Modeling'}\n")
            f.write(f"Test Perplexity: {test_ppl:.4f}\n\n")

            if model_type == "ssm":
                f.write("=== NWP Predictions (All Test Samples) ===\n")
                from tqdm import tqdm
                for inp, target in tqdm(test_loader, desc="Decoding NWP"):
                    inp = inp.to(device)
                    logits = model(inp)
                    pred_ids = logits.argmax(dim=-1)
                    for b in range(pred_ids.size(0)):
                        inp_str = decode_words(inp[b].tolist(), idx2word)
                        pred_str = decode_words(pred_ids[b].tolist(), idx2word)
                        tgt_str = decode_words(target[b].tolist(), idx2word)
                        f.write(f"Input    : {inp_str}\n")
                        f.write(f"Target   : {tgt_str}\n")
                        f.write(f"Predicted: {pred_str}\n")
                        f.write("-" * 80 + "\n")
            else:
                f.write("=== MLM Predictions (All Test Samples) ===\n")
                from tqdm import tqdm
                for masked_inp, original, mask_flags in tqdm(test_loader, desc="Decoding MLM"):
                    masked_inp = masked_inp.to(device)
                    logits = model(masked_inp)
                    pred_ids = logits.argmax(dim=-1)
                    for b in range(pred_ids.size(0)):
                        orig_str = decode_words(original[b].tolist(), idx2word)
                        pred_str = decode_words(pred_ids[b].tolist(), idx2word)
                        mask_str = decode_words(masked_inp[b].tolist(), idx2word)
                        f.write(f"Original : {orig_str}\n")
                        f.write(f"Masked   : {mask_str}\n")
                        f.write(f"Predicted: {pred_str}\n")
                        f.write("-" * 80 + "\n")

        print(f"Results saved → {result_path}")
        vocab_meta["test_perplexity"] = test_ppl

    # Save meta
    with open(meta_path, "w") as f:
        json.dump(vocab_meta, f)

    if use_wandb:
        try:
            finish_wandb()
        except Exception:
            pass

    print("Done.")
