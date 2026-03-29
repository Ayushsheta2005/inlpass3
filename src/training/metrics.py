"""
Evaluation metrics for all three tasks.

Task 1 (Decryption):
    - Character-level accuracy
    - Word-level accuracy
    - Levenshtein distance

Task 2 (Language Modeling):
    - Perplexity

Task 3 (Error correction):
    - All Task 1 metrics
    - BLEU score
    - ROUGE-{1,2,L} scores
"""

import math
from typing import List, Sequence


# ---------------------------------------------------------------------------
# Task 1 metrics
# ---------------------------------------------------------------------------

def char_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Character-level accuracy across a batch of string pairs."""
    correct = total = 0
    for pred, tgt in zip(predictions, targets):
        for p_ch, t_ch in zip(pred, tgt):
            correct += int(p_ch == t_ch)
            total += 1
    return correct / total if total > 0 else 0.0


def word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Word-level accuracy: fraction of words predicted exactly correctly."""
    correct = total = 0
    for pred, tgt in zip(predictions, targets):
        pred_words = pred.split()
        tgt_words = tgt.split()
        for pw, tw in zip(pred_words, tgt_words):
            correct += int(pw == tw)
            total += 1
    return correct / total if total > 0 else 0.0


def levenshtein(s: str, t: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    m, n = len(s), len(t)
    # dp[i][j] = edit distance between s[:i] and t[:j]
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s[i - 1] == t[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def mean_levenshtein(predictions: List[str], targets: List[str]) -> float:
    """Average Levenshtein distance across a list of string pairs."""
    if not predictions:
        return 0.0
    return sum(levenshtein(p, t) for p, t in zip(predictions, targets)) / len(predictions)


# ---------------------------------------------------------------------------
# Task 2 metrics
# ---------------------------------------------------------------------------

def perplexity(mean_nll: float) -> float:
    """Convert mean negative log-likelihood (nats) to perplexity."""
    return math.exp(mean_nll)


# ---------------------------------------------------------------------------
# Task 3 metrics: BLEU
# ---------------------------------------------------------------------------

def _ngram_counts(tokens: List[str], n: int) -> dict:
    counts: dict = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def bleu_score(
    predictions: List[str],
    targets: List[str],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU score (uniform weights for n=1..max_n).

    Uses a brevity penalty on the full corpus.

    Args:
        predictions: list of predicted strings (space-tokenised)
        targets:     list of reference strings (space-tokenised)
        max_n:       maximum n-gram order

    Returns:
        BLEU score in [0, 1]
    """
    import math

    pred_len = ref_len = 0
    clipped_counts = [0] * max_n
    total_counts   = [0] * max_n

    for pred_str, ref_str in zip(predictions, targets):
        pred_tokens = pred_str.split()
        ref_tokens  = ref_str.split()
        pred_len += len(pred_tokens)
        ref_len  += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_counts = _ngram_counts(pred_tokens, n)
            ref_counts  = _ngram_counts(ref_tokens, n)
            for gram, cnt in pred_counts.items():
                clipped_counts[n - 1] += min(cnt, ref_counts.get(gram, 0))
                total_counts[n - 1]   += cnt

    # Brevity penalty
    bp = 1.0 if pred_len >= ref_len else math.exp(1 - ref_len / max(pred_len, 1))

    log_bleu = 0.0
    for n in range(max_n):
        if clipped_counts[n] == 0 or total_counts[n] == 0:
            return 0.0
        log_bleu += math.log(clipped_counts[n] / total_counts[n])

    return bp * math.exp(log_bleu / max_n)


# ---------------------------------------------------------------------------
# Task 3 metrics: ROUGE
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Longest Common Subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_scores(
    predictions: List[str],
    targets: List[str],
) -> dict:
    """
    Compute corpus-level ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: list of predicted strings
        targets:     list of reference strings

    Returns:
        dict with keys 'rouge1', 'rouge2', 'rougeL' (float values in [0,1])
    """
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    r1_p = r1_r = 0.0
    r2_p = r2_r = 0.0
    rl_p = rl_r = 0.0
    n = len(predictions)

    for pred_str, ref_str in zip(predictions, targets):
        pred_toks = pred_str.split()
        ref_toks  = ref_str.split()
        len_pred  = max(len(pred_toks), 1)
        len_ref   = max(len(ref_toks), 1)

        # ROUGE-1
        pred_1 = _ngram_counts(pred_toks, 1)
        ref_1  = _ngram_counts(ref_toks,  1)
        overlap_1 = sum(min(cnt, ref_1.get(g, 0)) for g, cnt in pred_1.items())
        r1_p += overlap_1 / len_pred
        r1_r += overlap_1 / len_ref

        # ROUGE-2
        pred_2 = _ngram_counts(pred_toks, 2)
        ref_2  = _ngram_counts(ref_toks,  2)
        overlap_2 = sum(min(cnt, ref_2.get(g, 0)) for g, cnt in pred_2.items())
        r2_p += overlap_2 / max(len_pred - 1, 1)
        r2_r += overlap_2 / max(len_ref  - 1, 1)

        # ROUGE-L
        lcs = _lcs_length(pred_toks, ref_toks)
        rl_p += lcs / len_pred
        rl_r += lcs / len_ref

    return {
        "rouge1": _f1(r1_p / n, r1_r / n),
        "rouge2": _f1(r2_p / n, r2_r / n),
        "rougeL": _f1(rl_p / n, rl_r / n),
    }


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def compute_task1_metrics(
    predictions: List[str], targets: List[str]
) -> dict:
    return {
        "char_acc":   char_accuracy(predictions, targets),
        "word_acc":   word_accuracy(predictions, targets),
        "levenshtein": mean_levenshtein(predictions, targets),
    }


def compute_task3_metrics(
    predictions: List[str], targets: List[str]
) -> dict:
    m = compute_task1_metrics(predictions, targets)
    m["bleu"] = bleu_score(predictions, targets)
    m.update(rouge_scores(predictions, targets))
    return m
