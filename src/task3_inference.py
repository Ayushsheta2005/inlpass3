"""
Task 3: Language-Model-Assisted Decryption Error Correction
=============================================================
Loads a trained decryption model (RNN or LSTM from Task 1) and combines it
with each language model from Task 2 (SSM for NWP, Bi-LSTM for MLM) to
correct errors introduced by noise in the cipher transmission.

Correction strategies:
    SSM  – confidence-filtered word-level rescoring using NWP probabilities
    BiLSTM – mask low-confidence words and use MLM to fill them in

Experiments are run for each noisy cipher file (cipher_0{x}.txt).

Usage:
    python -m src.task3_inference --decrypt_model lstm \
        --decrypt_ckpt outputs/logs/task1_lstm_final.pt \
        --ssm_ckpt    outputs/logs/task2_ssm_final.pt   \
        --bilstm_ckpt outputs/logs/task2_bilstm_final.pt \
        --plain       data/plain.txt                     \
        --noisy_ciphers data/cipher_01.txt data/cipher_02.txt

Outputs:
    outputs/results/task3_{noise_level}_{system}.txt
    outputs/results/task3_summary.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import (
    build_char_vocab,
    build_word_vocab,
    clean_text,
    decode_chars,
    decode_words,
    encode_chars,
    encode_words,
    load_text,
)
from src.models.rnn import RNN
from src.models.lstm import LSTM
from src.models.ssm import S4Model
from src.models.bilstm import BiLSTM
from src.training.metrics import compute_task3_metrics
from src.utils.huggingface import load_checkpoint, pull_model
from src.utils.wandb_utils import finish_wandb, init_wandb, log_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Task 3 – Error Correction")
    p.add_argument("--decrypt_model", choices=["rnn", "lstm"], default="lstm",
                   help="Which decryption model to use from Task 1")
    p.add_argument("--decrypt_ckpt",  default="outputs/logs/task1_lstm_final.pt",
                   help="Path to local checkpoint for decryption model")
    p.add_argument("--decrypt_hf",    default=None,
                   help="HuggingFace repo for decryption model (alternative to local)")
    p.add_argument("--ssm_ckpt",      default="outputs/logs/task2_ssm_final.pt")
    p.add_argument("--ssm_hf",        default=None)
    p.add_argument("--bilstm_ckpt",   default="outputs/logs/task2_bilstm_final.pt")
    p.add_argument("--bilstm_hf",     default=None)
    p.add_argument("--plain",         default="data/plain.txt")
    p.add_argument("--noisy_ciphers", nargs="+",
                   default=["data/cipher_01.txt", "data/cipher_02.txt"])
    p.add_argument("--conf_threshold", type=float, default=0.5,
                   help="Per-character softmax confidence below which a token is "
                        "flagged as uncertain")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--seq_len",       type=int, default=256)
    p.add_argument("--lm_seq_len",    type=int, default=64)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--wandb_project", default="INLP_A3")
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--output_dir",    default="outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_decrypt_model(args, device):
    """Load decryption model from local checkpoint or HuggingFace."""
    if args.decrypt_hf:
        state_dict, meta = pull_model(args.decrypt_hf)
    else:
        state_dict, meta = load_checkpoint(args.decrypt_ckpt)

    cfg = meta["model_config"]
    char2idx = meta["char2idx"]
    idx2char = {int(k): v for k, v in meta["idx2char"].items()}

    if args.decrypt_model == "rnn":
        model = RNN(**cfg)
    else:
        model = LSTM(**cfg)

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"Decryption model ({args.decrypt_model.upper()}) loaded.")
    return model, char2idx, idx2char


def load_ssm_model(args, device):
    """Load S4 NWP model from local checkpoint or HuggingFace."""
    if args.ssm_hf:
        state_dict, meta = pull_model(args.ssm_hf)
    else:
        state_dict, meta = load_checkpoint(args.ssm_ckpt)

    cfg = meta["model_config"]
    word2idx = meta["word2idx"]
    idx2word = {int(k): v for k, v in meta["idx2word"].items()}

    model = S4Model(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["embed_dim"],
        N=cfg["N"],
        n_layers=cfg["num_layers"],
        l_max=cfg["seq_len"],
        dropout=0.0,
        pad_idx=word2idx["<PAD>"],
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("SSM (NWP) model loaded.")
    return model, word2idx, idx2word


def load_bilstm_model(args, device):
    """Load Bi-LSTM MLM model from local checkpoint or HuggingFace."""
    if args.bilstm_hf:
        state_dict, meta = pull_model(args.bilstm_hf)
    else:
        state_dict, meta = load_checkpoint(args.bilstm_ckpt)

    cfg = meta["model_config"]
    word2idx = meta["word2idx"]
    idx2word = {int(k): v for k, v in meta["idx2word"].items()}

    model = BiLSTM(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.0,
        pad_idx=word2idx["<PAD>"],
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("Bi-LSTM (MLM) model loaded.")
    return model, word2idx, idx2word


# ---------------------------------------------------------------------------
# Decryption pass (returns both text and per-char confidence)
# ---------------------------------------------------------------------------

@torch.no_grad()
def decrypt_with_confidence(
    decrypt_model,
    cipher_text: str,
    char2idx: Dict,
    idx2char: Dict,
    seq_len: int,
    batch_size: int,
    device,
) -> Tuple[str, List[float]]:
    """
    Run the decryption model on `cipher_text`.

    Returns:
        decrypted:   full decrypted string (greedy)
        confidence:  per-character softmax confidence scores
    """
    decrypt_model.eval()
    cipher_ids = encode_chars(cipher_text, char2idx)
    pad_id     = char2idx["<PAD>"]

    pred_chars: List[str] = []
    conf_scores: List[float] = []

    # Slide a window of seq_len over the cipher text
    for start in range(0, len(cipher_ids), seq_len):
        chunk = cipher_ids[start : start + seq_len]
        # Pad if shorter than seq_len
        pad_len = seq_len - len(chunk)
        chunk_padded = chunk + [pad_id] * pad_len

        x = torch.tensor([chunk_padded], dtype=torch.long, device=device)
        logits, _ = decrypt_model(x)           # (1, T, V)
        probs = F.softmax(logits[0], dim=-1)   # (T, V)

        for t in range(len(chunk)):   # only real (non-padded) positions
            best_id   = probs[t].argmax().item()
            best_conf = probs[t].max().item()
            pred_chars.append(idx2char.get(best_id, "?"))
            conf_scores.append(best_conf)

    return "".join(pred_chars), conf_scores


# ---------------------------------------------------------------------------
# Word-level confidence aggregation
# ---------------------------------------------------------------------------

def char_to_word_confidence(
    decrypted: str,
    char_conf: List[float],
) -> Tuple[List[str], List[float]]:
    """
    Aggregate character-level confidence to word level.

    Word confidence = min confidence among its characters (weakest-link).
    Returns word list and per-word minimum confidence.
    """
    words = decrypted.split()
    word_confs = []
    pos = 0

    for word in words:
        # Find the word in decrypted text starting at `pos`, skip spaces
        while pos < len(decrypted) and decrypted[pos] == " ":
            pos += 1
        end = pos + len(word)
        confs = char_conf[pos:end]
        word_confs.append(min(confs) if confs else 0.0)
        pos = end

    return words, word_confs


# ---------------------------------------------------------------------------
# SSM-based correction (NWP rescoring)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ssm_correct(
    words: List[str],
    word_confs: List[float],
    ssm_model,
    word2idx: Dict,
    idx2word: Dict,
    conf_threshold: float,
    lm_seq_len: int,
    device,
) -> List[str]:
    """
    Replace low-confidence words using autoregressive SSM prediction.

    For each low-confidence position t:
        Feed the context words[0..t-1] into the SSM,
        replace words[t] with the SSM's top-1 next-word prediction.

    Args:
        words:           decoded word list from decryption model
        word_confs:      per-word confidence scores
        conf_threshold:  words below this confidence are flagged

    Returns:
        corrected word list
    """
    ssm_model.eval()
    corrected = words[:]
    pad_id    = word2idx.get("<PAD>", 0)
    unk_id    = word2idx.get("<UNK>", 3)

    for t, conf in enumerate(word_confs):
        if conf >= conf_threshold:
            continue  # word is confident, keep it

        # Build context: last lm_seq_len words before position t
        ctx_start = max(0, t - lm_seq_len)
        ctx_words  = corrected[ctx_start:t]
        ctx_ids    = [word2idx.get(w, unk_id) for w in ctx_words]

        if not ctx_ids:
            continue

        # Pad or truncate to lm_seq_len
        if len(ctx_ids) < lm_seq_len:
            ctx_ids = [pad_id] * (lm_seq_len - len(ctx_ids)) + ctx_ids
        else:
            ctx_ids = ctx_ids[-lm_seq_len:]

        x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        logits = ssm_model(x)                     # (1, T, V)
        pred_id = logits[0, -1].argmax().item()   # next-word prediction
        pred_word = idx2word.get(pred_id, words[t])

        corrected[t] = pred_word

    return corrected


# ---------------------------------------------------------------------------
# Bi-LSTM-based correction (MLM filling)
# ---------------------------------------------------------------------------

@torch.no_grad()
def bilstm_correct(
    words: List[str],
    word_confs: List[float],
    bilstm_model,
    word2idx: Dict,
    idx2word: Dict,
    conf_threshold: float,
    lm_seq_len: int,
    device,
) -> List[str]:
    """
    Replace low-confidence words using Bi-LSTM MLM prediction.

    Processes the sequence in windows of lm_seq_len.  Within each window,
    low-confidence positions are replaced with <MASK> and the Bi-LSTM
    predicts the original tokens.

    Args:
        words:           decoded word list
        word_confs:      per-word confidence scores
        conf_threshold:  masking threshold

    Returns:
        corrected word list
    """
    bilstm_model.eval()
    corrected  = words[:]
    pad_id     = word2idx.get("<PAD>", 0)
    mask_id    = word2idx.get("<MASK>", 4)
    unk_id     = word2idx.get("<UNK>", 3)
    n          = len(words)

    for start in range(0, n, lm_seq_len):
        chunk_words = words[start : start + lm_seq_len]
        chunk_confs = word_confs[start : start + lm_seq_len]

        chunk_ids   = [word2idx.get(w, unk_id) for w in chunk_words]
        masked_ids  = chunk_ids[:]
        mask_flags  = []

        for i, conf in enumerate(chunk_confs):
            if conf < conf_threshold:
                masked_ids[i] = mask_id
                mask_flags.append(1)
            else:
                mask_flags.append(0)

        # Pad to lm_seq_len
        pad_len = lm_seq_len - len(chunk_ids)
        padded  = masked_ids + [pad_id] * pad_len

        x      = torch.tensor([padded],     dtype=torch.long, device=device)
        logits = bilstm_model(x)            # (1, lm_seq_len, V)
        preds  = logits[0].argmax(dim=-1)   # (lm_seq_len,)

        for i, (flag, w) in enumerate(zip(mask_flags, chunk_words)):
            if flag:
                pred_word = idx2word.get(preds[i].item(), w)
                corrected[start + i] = pred_word

    return corrected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(f"{args.output_dir}/results", exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        init_wandb(
            project=args.wandb_project,
            run_name="task3_error_correction",
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    decrypt_model, char2idx, idx2char = load_decrypt_model(args, device)
    ssm_model,     word2idx_ssm,  idx2word_ssm  = load_ssm_model(args, device)
    bilstm_model,  word2idx_blstm, idx2word_blstm = load_bilstm_model(args, device)

    # ------------------------------------------------------------------
    # Load plain text (ground truth)
    # ------------------------------------------------------------------
    plain_text = clean_text(load_text(args.plain))

    summary: Dict = {}

    # ------------------------------------------------------------------
    # Iterate over noisy cipher files
    # ------------------------------------------------------------------
    for cipher_path in args.noisy_ciphers:
        noise_tag = os.path.splitext(os.path.basename(cipher_path))[0]  # e.g. cipher_01
        print(f"\n{'='*60}")
        print(f"Processing: {cipher_path}  (tag={noise_tag})")
        print(f"{'='*60}")

        cipher_text = clean_text(load_text(cipher_path))

        # -----------------------------------------------------------
        # Align plain text length to cipher text length
        # -----------------------------------------------------------
        min_len = min(len(cipher_text), len(plain_text))
        cipher_text = cipher_text[:min_len]
        ref_text    = plain_text[:min_len]

        # -----------------------------------------------------------
        # 1. Decryption alone
        # -----------------------------------------------------------
        decrypted_raw, char_conf = decrypt_with_confidence(
            decrypt_model, cipher_text, char2idx, idx2char,
            args.seq_len, args.batch_size, device,
        )
        decrypt_words, word_confs = char_to_word_confidence(decrypted_raw, char_conf)
        ref_words = ref_text.split()

        align = min(len(decrypt_words), len(ref_words))
        decrypt_words = decrypt_words[:align]
        word_confs    = word_confs[:align]
        ref_words     = ref_words[:align]

        decrypt_str = " ".join(decrypt_words)
        ref_str     = " ".join(ref_words)

        m_decrypt = compute_task3_metrics([decrypt_str], [ref_str])
        print(f"[Decrypt only] {m_decrypt}")

        _save_result(
            f"{args.output_dir}/results/task3_{noise_tag}_decrypt.txt",
            "Decryption Only", cipher_text[:500], decrypt_str[:500], ref_str[:500],
            m_decrypt,
        )

        # -----------------------------------------------------------
        # 2. Decryption + SSM
        # -----------------------------------------------------------
        ssm_corrected = ssm_correct(
            decrypt_words, word_confs, ssm_model,
            word2idx_ssm, idx2word_ssm, args.conf_threshold,
            args.lm_seq_len, device,
        )
        ssm_str = " ".join(ssm_corrected)
        m_ssm   = compute_task3_metrics([ssm_str], [ref_str])
        print(f"[Decrypt + SSM] {m_ssm}")

        _save_result(
            f"{args.output_dir}/results/task3_{noise_tag}_ssm.txt",
            "Decryption + SSM", cipher_text[:500], ssm_str[:500], ref_str[:500],
            m_ssm,
        )

        # -----------------------------------------------------------
        # 3. Decryption + Bi-LSTM
        # -----------------------------------------------------------
        bilstm_corrected = bilstm_correct(
            decrypt_words, word_confs, bilstm_model,
            word2idx_blstm, idx2word_blstm, args.conf_threshold,
            args.lm_seq_len, device,
        )
        bilstm_str = " ".join(bilstm_corrected)
        m_bilstm   = compute_task3_metrics([bilstm_str], [ref_str])
        print(f"[Decrypt + BiLSTM] {m_bilstm}")

        _save_result(
            f"{args.output_dir}/results/task3_{noise_tag}_bilstm.txt",
            "Decryption + Bi-LSTM", cipher_text[:500], bilstm_str[:500], ref_str[:500],
            m_bilstm,
        )

        # -----------------------------------------------------------
        # Log to WandB
        # -----------------------------------------------------------
        if use_wandb:
            log_metrics(
                {
                    f"{noise_tag}/decrypt_{k}": v
                    for k, v in m_decrypt.items()
                }
            )
            log_metrics(
                {f"{noise_tag}/ssm_{k}": v for k, v in m_ssm.items()}
            )
            log_metrics(
                {f"{noise_tag}/bilstm_{k}": v for k, v in m_bilstm.items()}
            )

        summary[noise_tag] = {
            "decrypt": m_decrypt,
            "decrypt_ssm": m_ssm,
            "decrypt_bilstm": m_bilstm,
        }

    # ------------------------------------------------------------------
    # Write summary
    # ------------------------------------------------------------------
    summary_path = f"{args.output_dir}/results/task3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    _print_summary_table(summary)

    finish_wandb()
    print("Done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_result(path, system_name, cipher_sample, pred_sample, ref_sample, metrics):
    with open(path, "w") as f:
        f.write(f"System: {system_name}\n\n")
        f.write(f"Input (cipher, truncated):\n{cipher_sample}\n\n")
        f.write(f"Reference (plain):\n{ref_sample}\n\n")
        f.write(f"Prediction:\n{pred_sample}\n\n")
        f.write("Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")


def _print_summary_table(summary: Dict):
    print("\n" + "=" * 80)
    print(f"{'Noise':<15} {'System':<22} {'CharAcc':>8} {'WordAcc':>8} {'Lev':>8} {'BLEU':>8}")
    print("-" * 80)
    for noise_tag, systems in summary.items():
        for sys_name, m in systems.items():
            print(
                f"{noise_tag:<15} {sys_name:<22} "
                f"{m.get('char_acc', 0):.4f}   "
                f"{m.get('word_acc', 0):.4f}   "
                f"{m.get('levenshtein', 0):7.1f}   "
                f"{m.get('bleu', 0):.4f}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
