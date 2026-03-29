"""
Data preprocessing utilities for Assignment 3.
Handles loading cipher/plain text files and building vocabularies.
"""

import re
import os
from typing import Tuple, List, Dict


def load_text(path: str) -> str:
    """Load a text file and return its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Lowercase and strip excess whitespace. Preserve all characters."""
    text = text.lower()
    text = re.sub(r"\r\n|\r", "\n", text)  # normalise line endings
    return text


def build_char_vocab(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build character-level vocabulary from a list of texts.

    Returns:
        char2idx: mapping character → index
        idx2char: mapping index → character
    """
    chars = sorted(set("".join(texts)))
    # Reserve index 0 for <PAD>, 1 for <SOS>, 2 for <EOS>, 3 for <UNK>
    special = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    vocab = special + chars
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}
    return char2idx, idx2char


def build_word_vocab(texts: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build word-level vocabulary from a list of texts.

    Args:
        texts:    list of raw text strings
        min_freq: minimum token frequency; words seen fewer times become <UNK>

    Returns:
        word2idx: mapping word -> index
        idx2word: mapping index -> word
    """
    from collections import Counter
    counter = Counter(tok for text in texts for tok in text.split())
    words = sorted(w for w, c in counter.items() if c >= min_freq)
    special = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<MASK>"]
    vocab = special + words
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word


def encode_chars(text: str, char2idx: Dict[str, int]) -> List[int]:
    """Encode a string into a list of character indices."""
    unk = char2idx["<UNK>"]
    return [char2idx.get(ch, unk) for ch in text]


def decode_chars(indices: List[int], idx2char: Dict[int, str]) -> str:
    """Decode a list of character indices back to a string."""
    specials = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}
    return "".join(
        idx2char[i] for i in indices if idx2char.get(i, "") not in specials
    )


def encode_words(text: str, word2idx: Dict[str, int]) -> List[int]:
    """Encode whitespace-tokenised text into a list of word indices."""
    unk = word2idx["<UNK>"]
    return [word2idx.get(w, unk) for w in text.split()]


def decode_words(indices: List[int], idx2word: Dict[int, str]) -> str:
    """Decode a list of word indices back to a string."""
    specials = {"<PAD>", "<SOS>", "<EOS>", "<UNK>", "<MASK>"}
    return " ".join(
        idx2word[i] for i in indices if idx2word.get(i, "") not in specials
    )


def split_data(
    data: List,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List, List, List]:
    """
    Split data into train / validation / test sets deterministically.

    Args:
        data: list of samples
        train_ratio: fraction of data for training
        val_ratio: fraction of data for validation (remainder goes to test)

    Returns:
        train, val, test splits
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]


# Integer token used to pad cipher sequences (values 0-99 are real tokens).
CIPHER_PAD = 100


def tokenize_cipher_line(c_str: str, p_str: str) -> List[int]:
    """Decode a cipher line into integer tokens guided by the plain text.

    Encoding rule:
      - space character → 1-digit decimal code (0–9)
      - any other character → 2-digit decimal code (00–99)

    Args:
        c_str: raw cipher line (string of decimal digits)
        p_str: corresponding plain text line (used as a decoding guide)

    Returns:
        List of integer tokens, one per plain character.
    """
    c = c_str.strip()
    p = p_str.strip().lower()
    tokens: List[int] = []
    i = 0
    for ch in p:
        if i >= len(c):
            break
        if ch == " ":
            tokens.append(int(c[i]))
            i += 1
        else:
            if i + 1 >= len(c):
                break
            tokens.append(int(c[i : i + 2]))
            i += 2
    return tokens


def build_parallel_samples(
    cipher_text: str,
    plain_text: str,
    char2idx: Dict[str, int],
    seq_len: int = 256,
) -> List[Tuple[List[int], List[int]]]:
    """
    Build aligned (cipher_tokens, plain_ids) sample pairs of fixed length.

    The cipher is decoded line-by-line using the known encoding:
      - space  → 1-digit decimal token (0–9)
      - letter → 2-digit decimal token (00–99)

    Cipher tokens are integers in [0, 99]; cipher padding uses CIPHER_PAD (100).
    Plain tokens are char indices from char2idx; plain padding uses char2idx["<PAD>"].

    Args:
        cipher_text: full cipher file text (newline-separated lines of digits)
        plain_text:  full plain file text (newline-separated lines)
        char2idx:    character vocabulary for plain text
        seq_len:     fixed sequence length (truncate or pad to this value)

    Returns:
        List of (cipher_ids, plain_ids) tuples, each of length seq_len.
    """
    plain_pad = char2idx["<PAD>"]
    unk_id = char2idx.get("<UNK>", 3)

    def _pad_cipher(ids: List[int], length: int) -> List[int]:
        if len(ids) >= length:
            return ids[:length]
        return ids + [CIPHER_PAD] * (length - len(ids))

    def _pad_plain(ids: List[int], length: int) -> List[int]:
        if len(ids) >= length:
            return ids[:length]
        return ids + [plain_pad] * (length - len(ids))

    cipher_lines = cipher_text.splitlines()
    plain_lines = plain_text.splitlines()

    samples = []
    for c_line, p_line in zip(cipher_lines, plain_lines):
        if not c_line.strip() or not p_line.strip():
            continue
        # Decode cipher as integer tokens (space→1 digit, letter→2 digits)
        c_ids = tokenize_cipher_line(c_line, p_line)
        # Encode plain as char indices (lower-case, guided by clean_text)
        p_ids = [char2idx.get(ch, unk_id) for ch in p_line.strip().lower()]
        samples.append(
            (_pad_cipher(c_ids, seq_len), _pad_plain(p_ids, seq_len))
        )
    return samples


def build_lm_samples(
    text: str,
    word2idx: Dict[str, int],
    seq_len: int = 64,
) -> List[Tuple[List[int], List[int]]]:
    """
    Build (input, target) pairs for Next Word Prediction.

    Slides a window: input = tokens[i:i+seq_len], target = tokens[i+1:i+seq_len+1].

    Returns:
        List of (input_ids, target_ids) tuples.
    """
    token_ids = encode_words(text, word2idx)
    samples = []
    for start in range(0, len(token_ids) - seq_len, seq_len):
        inp = token_ids[start : start + seq_len]
        tgt = token_ids[start + 1 : start + seq_len + 1]
        samples.append((inp, tgt))
    return samples


def build_mlm_samples(
    text: str,
    word2idx: Dict[str, int],
    seq_len: int = 64,
    mask_prob: float = 0.15,
    seed: int = 42,
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Build (masked_input, target, mask_positions) triples for Masked LM.

    Implements the BERT 80-10-10 rule for the mask_prob fraction of tokens:
      - 80% of selected tokens → replaced with <MASK>
      - 10% of selected tokens → replaced with a random vocab word
      - 10% of selected tokens → kept as-is (but still counted as masked)

    Returns:
        List of (masked_ids, original_ids, mask_flags) tuples.
    """
    import random

    random.seed(seed)
    mask_id = word2idx["<MASK>"]
    # Build normal (non-special) vocab for random replacement
    special_ids = {word2idx[s] for s in ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<MASK>"] if s in word2idx}
    normal_vocab = [i for i in range(len(word2idx)) if i not in special_ids]
    token_ids = encode_words(text, word2idx)

    samples = []
    for start in range(0, len(token_ids) - seq_len + 1, seq_len):
        original = token_ids[start : start + seq_len]
        masked = original[:]
        mask_flags = [0] * seq_len
        for i in range(seq_len):
            if random.random() < mask_prob:
                r = random.random()
                if r < 0.8:
                    masked[i] = mask_id          # 80% → [MASK]
                elif r < 0.9:
                    masked[i] = random.choice(normal_vocab)  # 10% → random word
                # else: keep original (10%)
                mask_flags[i] = 1
        samples.append((masked, original, mask_flags))
    return samples
