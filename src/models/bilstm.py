"""
Bidirectional LSTM implemented from scratch using only basic PyTorch operations.
nn.RNN and nn.LSTM are NOT used.

Architecture:
    Embedding → [Forward LSTM ‖ Backward LSTM] → Concat → Linear → logits

Reference: Anishnama (2023) "Understanding Bidirectional LSTM for Sequential Data Processing"

The Bi-LSTM is used for Masked Language Modeling (MLM):
    - Input tokens with some replaced by <MASK>
    - Model predicts the original token at each masked position
    - At inference, only masked positions are evaluated for perplexity
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .lstm import LSTMCell


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for Masked Language Modeling.

    Uses two independent stacks of LSTMCells:
        • forward stack  : processes sequence left → right
        • backward stack : processes sequence right → left

    At each position t the hidden states from both directions are concatenated:
        h_t = [h_fwd_t ; h_bwd_t]   (2 * hidden_size)

    Args:
        vocab_size:   vocabulary size
        embed_dim:    embedding dimensionality
        hidden_size:  hidden size per direction (output is 2 * hidden_size)
        num_layers:   stacked layers per direction
        dropout:      inter-layer dropout probability
        pad_idx:      padding token index
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # Forward direction cells
        self.fwd_cells = nn.ModuleList()
        # Backward direction cells
        self.bwd_cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = embed_dim if layer == 0 else hidden_size
            self.fwd_cells.append(LSTMCell(in_size, hidden_size))
            self.bwd_cells.append(LSTMCell(in_size, hidden_size))

        # Projection from concatenated bidirectional hidden state → vocab
        self.output_proj = nn.Linear(2 * hidden_size, vocab_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_direction(
        self,
        embeds: torch.Tensor,
        cells: nn.ModuleList,
        reverse: bool,
    ) -> torch.Tensor:
        """Run a stack of LSTMCells over `embeds` in one direction.

        Args:
            embeds:  (batch, seq_len, embed_dim)
            cells:   ModuleList of LSTMCells (one per layer)
            reverse: if True, iterate time steps in reverse order

        Returns:
            outputs: (batch, seq_len, hidden_size) — top-layer hidden states
        """
        batch, seq_len, _ = embeds.shape
        device = embeds.device

        # Initialise hidden/cell states for each layer
        h = [
            torch.zeros(batch, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(batch, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        outputs = [None] * seq_len  # type: ignore[list-item]

        for t in time_steps:
            inp = embeds[:, t, :]  # (B, E)
            for layer_idx, cell in enumerate(cells):
                h_new, c_new = cell(inp, (h[layer_idx], c[layer_idx]))
                inp = (
                    self.dropout(h_new)
                    if layer_idx < self.num_layers - 1
                    else h_new
                )
                h[layer_idx] = h_new
                c[layer_idx] = c_new
            outputs[t] = inp  # top-layer output at time t

        return torch.stack(outputs, dim=1)  # (B, T, H)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer token indices (some may be <MASK>)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        embeds = self.dropout(self.embedding(x))  # (B, T, E)

        fwd_out = self._run_direction(embeds, self.fwd_cells, reverse=False)
        bwd_out = self._run_direction(embeds, self.bwd_cells, reverse=True)

        combined = torch.cat([fwd_out, bwd_out], dim=-1)  # (B, T, 2*H)
        logits = self.output_proj(combined)               # (B, T, vocab_size)
        return logits
