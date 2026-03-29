"""
Task 3 Pipeline – LM-Assisted Cipher Decryption Error Correction
=================================================================
Loads Task-1 decryption model + Task-2 language models to correct
low-confidence words in the decrypted output.

Called by both task3_ssm and task3_bilstm subcommands via main.py.
The correction_model key in the config controls which LM is applied.

Usage (via main.py):
    uv run main.py task3_ssm --mode evaluate
    uv run main.py task3_bilstm --mode evaluate
"""

import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml

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
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_state_dict(ckpt_path: str, device):
    return torch.load(ckpt_path, map_location=device, weights_only=True)


def _load_meta(meta_path: str) -> dict:
    with open(meta_path) as f:
        return json.load(f)


def load_decrypt_model(cfg: dict, device):
    """Load Task-1 decryption model (RNN or LSTM)."""
    model_type = cfg["task1_model"]
    ckpt_path = cfg["task1_ckpt"]
    meta_path = cfg["task1_meta"]

    meta = _load_meta(meta_path)
    char2idx = meta["char2idx"]
    idx2char = {int(k): v for k, v in meta["idx2char"].items()}
    model_cfg = meta["model_config"]

    model = (RNN if model_type == "rnn" else LSTM)(**model_cfg)
    model.load_state_dict(_load_state_dict(ckpt_path, device))
    model = model.to(device).eval()
    print(f"Loaded Task-1 model ({model_type.upper()}) from {ckpt_path}")
    return model, char2idx, idx2char


def load_ssm_model(cfg: dict, device):
    """Load Task-2 SSM (NWP) model."""
    ckpt_path = cfg["task2_ssm_ckpt"]
    meta_path = cfg["task2_ssm_meta"]

    meta = _load_meta(meta_path)
    word2idx = meta["word2idx"]
    idx2word = {int(k): v for k, v in meta["idx2word"].items()}
    mc = meta["model_config"]

    model = S4Model(
        vocab_size=mc["vocab_size"],
        d_model=mc["embed_dim"],
        N=mc["N"],
        n_layers=mc["num_layers"],
        l_max=mc["seq_len"],
        dropout=0.0,
        pad_idx=word2idx["<PAD>"],
    )
    model.load_state_dict(_load_state_dict(ckpt_path, device))
    model = model.to(device).eval()
    print(f"Loaded SSM model from {ckpt_path}")
    return model, word2idx, idx2word


def load_bilstm_model(cfg: dict, device):
    """Load Task-2 BiLSTM (MLM) model."""
    ckpt_path = cfg["task2_bilstm_ckpt"]
    meta_path = cfg["task2_bilstm_meta"]

    meta = _load_meta(meta_path)
    word2idx = meta["word2idx"]
    idx2word = {int(k): v for k, v in meta["idx2word"].items()}
    mc = meta["model_config"]

    model = BiLSTM(
        vocab_size=mc["vocab_size"],
        embed_dim=mc["embed_dim"],
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=0.0,
        pad_idx=word2idx["<PAD>"],
    )
    model.load_state_dict(_load_state_dict(ckpt_path, device))
    model = model.to(device).eval()
    print(f"Loaded BiLSTM model from {ckpt_path}")
    return model, word2idx, idx2word


# ---------------------------------------------------------------------------
# Decryption with confidence
# ---------------------------------------------------------------------------

@torch.no_grad()
def decrypt_with_confidence(
    model, cipher_text: str, char2idx: dict, idx2char: dict,
    seq_len: int, device
) -> Tuple[str, List[float]]:
    """Greedy decryption returning predicted text + per-char confidence."""
    model.eval()
    cipher_ids = encode_chars(cipher_text, char2idx)
    pad_id = char2idx["<PAD>"]
    pred_chars: List[str] = []
    conf_scores: List[float] = []

    for start in range(0, len(cipher_ids), seq_len):
        chunk = cipher_ids[start: start + seq_len]
        pad_len = seq_len - len(chunk)
        padded = chunk + [pad_id] * pad_len
        x = torch.tensor([padded], dtype=torch.long, device=device)
        logits, _ = model(x)            # (1, T, V)
        probs = F.softmax(logits[0], dim=-1)   # (T, V)
        for t in range(len(chunk)):
            best_id = probs[t].argmax().item()
            best_conf = probs[t].max().item()
            pred_chars.append(idx2char.get(best_id, "?"))
            conf_scores.append(best_conf)

    return "".join(pred_chars), conf_scores


# ---------------------------------------------------------------------------
# Word-level confidence
# ---------------------------------------------------------------------------

def char_to_word_confidence(
    decrypted: str, char_conf: List[float]
) -> Tuple[List[str], List[float]]:
    words = decrypted.split()
    word_confs: List[float] = []
    pos = 0
    for word in words:
        while pos < len(decrypted) and decrypted[pos] == " ":
            pos += 1
        end = pos + len(word)
        confs = char_conf[pos:end]
        word_confs.append(min(confs) if confs else 0.0)
        pos = end
    return words, word_confs


# ---------------------------------------------------------------------------
# SSM correction (NWP-based)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ssm_correct(
    words: List[str], word_confs: List[float],
    ssm_model, word2idx: dict, idx2word: dict,
    conf_threshold: float, lm_seq_len: int, device
) -> List[str]:
    ssm_model.eval()
    corrected = words[:]
    pad_id = word2idx.get("<PAD>", 0)
    unk_id = word2idx.get("<UNK>", 3)

    for t, conf in enumerate(word_confs):
        if conf >= conf_threshold:
            continue
        ctx_start = max(0, t - lm_seq_len)
        ctx_ids = [word2idx.get(w, unk_id) for w in corrected[ctx_start:t]]
        if not ctx_ids:
            continue
        if len(ctx_ids) < lm_seq_len:
            ctx_ids = [pad_id] * (lm_seq_len - len(ctx_ids)) + ctx_ids
        else:
            ctx_ids = ctx_ids[-lm_seq_len:]
        x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        logits = ssm_model(x)
        pred_id = logits[0, -1].argmax().item()
        corrected[t] = idx2word.get(pred_id, words[t])

    return corrected


# ---------------------------------------------------------------------------
# BiLSTM correction (MLM-based)
# ---------------------------------------------------------------------------

@torch.no_grad()
def bilstm_correct(
    words: List[str], word_confs: List[float],
    bilstm_model, word2idx: dict, idx2word: dict,
    conf_threshold: float, lm_seq_len: int, device
) -> List[str]:
    bilstm_model.eval()
    corrected = words[:]
    pad_id = word2idx.get("<PAD>", 0)
    mask_id = word2idx.get("<MASK>", 4)
    unk_id = word2idx.get("<UNK>", 3)

    for start in range(0, len(words), lm_seq_len):
        chunk_words = words[start: start + lm_seq_len]
        chunk_confs = word_confs[start: start + lm_seq_len]
        chunk_ids = [word2idx.get(w, unk_id) for w in chunk_words]
        masked_ids = chunk_ids[:]
        mask_flags = []
        for i, conf in enumerate(chunk_confs):
            if conf < conf_threshold:
                masked_ids[i] = mask_id
                mask_flags.append(1)
            else:
                mask_flags.append(0)
        pad_len = lm_seq_len - len(chunk_ids)
        padded = masked_ids + [pad_id] * pad_len
        x = torch.tensor([padded], dtype=torch.long, device=device)
        logits = bilstm_model(x)
        preds = logits[0].argmax(dim=-1)
        for i, (flag, w) in enumerate(zip(mask_flags, chunk_words)):
            if flag:
                corrected[start + i] = idx2word.get(preds[i].item(), w)

    return corrected


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _save_result(path: str, system_name: str, cipher_sample: str,
                 pred_sample: str, ref_sample: str, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"System: {system_name}\n\n")
        f.write(f"Input (cipher, truncated):\n{cipher_sample}\n\n")
        f.write(f"Reference:\n{ref_sample}\n\n")
        f.write(f"Prediction:\n{pred_sample}\n\n")
        f.write("Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")


def _print_summary_table(summary: dict):
    print("\n" + "=" * 80)
    print(f"{'Noise':<15} {'System':<22} {'CharAcc':>8} {'WordAcc':>8} {'BLEU':>8}")
    print("-" * 80)
    for noise_tag, systems in summary.items():
        for sys_name, m in systems.items():
            print(
                f"{noise_tag:<15} {sys_name:<22} "
                f"{m.get('char_acc', 0):.4f}   "
                f"{m.get('word_acc', 0):.4f}   "
                f"{m.get('bleu', 0):.4f}"
            )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config_path: str, mode: str = "evaluate"):
    """Entry point called by main.py for task3_ssm and task3_bilstm."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if mode == "train":
        print(
            "Task 3 has no standalone training step.\n"
            "Run task1_rnn/task1_lstm + task2_ssm/task2_bilstm first, "
            "then re-run task3 with --mode evaluate."
        )
        return

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correction_model = cfg.get("correction_model", "ssm")
    print(f"[Task 3 | correction={correction_model.upper()}] device={device}")

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(f"{out_dir}/results", exist_ok=True)

    use_wandb = not cfg.get("no_wandb", False)
    if use_wandb:
        try:
            init_wandb(project=cfg.get("wandb_project", "INLP_A3"),
                       config=cfg, name=f"task3_{correction_model}")
        except Exception:
            use_wandb = False

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    decrypt_model, char2idx, idx2char = load_decrypt_model(cfg, device)
    ssm_model, word2idx_ssm, idx2word_ssm = load_ssm_model(cfg, device)
    bilstm_model, word2idx_blstm, idx2word_blstm = load_bilstm_model(cfg, device)

    # Load ground-truth plain text
    plain_text = clean_text(load_text(cfg["plain"]))

    seq_len = cfg.get("seq_len", 256)
    lm_seq_len = cfg.get("lm_seq_len", 64)
    conf_threshold = cfg.get("conf_threshold", 0.5)
    summary: Dict = {}

    # ------------------------------------------------------------------
    # Iterate over cipher files
    # ------------------------------------------------------------------
    cipher_files = cfg.get("cipher_files", [])
    for cipher_path in cipher_files:
        noise_tag = os.path.splitext(os.path.basename(cipher_path))[0]
        print(f"\n{'='*60}")
        print(f"Processing: {cipher_path}  (tag={noise_tag})")
        print(f"{'='*60}")

        cipher_text = clean_text(load_text(cipher_path))
        min_len = min(len(cipher_text), len(plain_text))
        cipher_text = cipher_text[:min_len]
        ref_text = plain_text[:min_len]

        # Step 1: Decrypt with confidence
        decrypted_raw, char_conf = decrypt_with_confidence(
            decrypt_model, cipher_text, char2idx, idx2char, seq_len, device
        )
        decrypt_words_list, word_confs = char_to_word_confidence(decrypted_raw, char_conf)
        ref_words = ref_text.split()

        align = min(len(decrypt_words_list), len(ref_words))
        decrypt_words_list = decrypt_words_list[:align]
        word_confs = word_confs[:align]
        ref_words = ref_words[:align]

        decrypt_str = " ".join(decrypt_words_list)
        ref_str = " ".join(ref_words)

        m_decrypt = compute_task3_metrics([decrypt_str], [ref_str])
        print(f"[Decrypt only]  {m_decrypt}")
        _save_result(
            f"{out_dir}/results/task3_{noise_tag}_decrypt.txt",
            "Decryption Only", cipher_text[:500], decrypt_str[:500], ref_str[:500],
            m_decrypt,
        )

        # Step 2: Apply SSM correction
        ssm_corrected = ssm_correct(
            decrypt_words_list, word_confs, ssm_model,
            word2idx_ssm, idx2word_ssm, conf_threshold, lm_seq_len, device
        )
        ssm_str = " ".join(ssm_corrected)
        m_ssm = compute_task3_metrics([ssm_str], [ref_str])
        print(f"[Decrypt + SSM] {m_ssm}")
        _save_result(
            f"{out_dir}/results/task3_{noise_tag}_ssm.txt",
            "Decryption + SSM", cipher_text[:500], ssm_str[:500], ref_str[:500],
            m_ssm,
        )

        # Step 3: Apply BiLSTM correction
        bilstm_corrected = bilstm_correct(
            decrypt_words_list, word_confs, bilstm_model,
            word2idx_blstm, idx2word_blstm, conf_threshold, lm_seq_len, device
        )
        bilstm_str = " ".join(bilstm_corrected)
        m_bilstm = compute_task3_metrics([bilstm_str], [ref_str])
        print(f"[Decrypt + BiLSTM] {m_bilstm}")
        _save_result(
            f"{out_dir}/results/task3_{noise_tag}_bilstm.txt",
            "Decryption + BiLSTM", cipher_text[:500], bilstm_str[:500], ref_str[:500],
            m_bilstm,
        )

        if use_wandb:
            try:
                log_wandb({f"{noise_tag}/decrypt_{k}": v for k, v in m_decrypt.items()})
                log_wandb({f"{noise_tag}/ssm_{k}": v for k, v in m_ssm.items()})
                log_wandb({f"{noise_tag}/bilstm_{k}": v for k, v in m_bilstm.items()})
            except Exception:
                pass

        summary[noise_tag] = {
            "decrypt": m_decrypt,
            "decrypt_ssm": m_ssm,
            "decrypt_bilstm": m_bilstm,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_path = f"{out_dir}/results/task3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    _print_summary_table(summary)

    if use_wandb:
        try:
            finish_wandb()
        except Exception:
            pass

    print("Done.")
