"""
Vanilla RNN implemented from scratch using only basic PyTorch operations.
nn.RNN and nn.LSTM are NOT used.

Architecture:
    Embedding → RNN cells (stacked) → Linear projection → logits

The RNN cell update rule:
    h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RNNCell(nn.Module):
    """Single-step vanilla RNN cell.

    Args:
        input_size:  dimensionality of input vector
        hidden_size: dimensionality of hidden state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Input → hidden weights and bias
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        # Hidden → hidden weights and bias
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights similar to standard PyTorch nn.RNN (1/sqrt(hidden_size))."""
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in [self.W_ih, self.W_hh, self.b_ih, self.b_hh]:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:      (batch, input_size)
            h_prev: (batch, hidden_size)

        Returns:
            h_t:    (batch, hidden_size)
        """
        h_t = torch.tanh(
            x @ self.W_ih.t() + self.b_ih
            + h_prev @ self.W_hh.t() + self.b_hh
        )
        return h_t


class RNN(nn.Module):
    """
    Multi-layer stacked RNN for sequence-to-sequence character-level decryption.

    Args:
        vocab_size:        size of the *output* (plain) character vocabulary
        embed_dim:         embedding dimensionality
        hidden_size:       RNN hidden state size
        num_layers:        number of stacked RNN layers
        dropout:           dropout probability between layers (applied to outputs of
                           all layers except the last)
        pad_idx:           index used for padding in the *output* vocab
        cipher_vocab_size: size of the *input* (cipher) token vocabulary
                           (default 101: integer tokens 0-99 + pad=100)
        cipher_pad_idx:    padding index for cipher input sequences (default 100)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        cipher_vocab_size: int = 101,
        cipher_pad_idx: int = 100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input embedding: cipher tokens (0-99 real, 100 = pad)
        self.embedding = nn.Embedding(
            cipher_vocab_size, embed_dim, padding_idx=cipher_pad_idx
        )

        # One RNNCell per layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = embed_dim if layer == 0 else hidden_size
            self.cells.append(RNNCell(in_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        # Output projection: plain character vocab
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:   (batch, seq_len) integer token indices
            h_0: (num_layers, batch, hidden_size) initial hidden states,
                 or None to default to zeros

        Returns:
            logits: (batch, seq_len, vocab_size)
            h_n:    (num_layers, batch, hidden_size) final hidden states
        """
        batch, seq_len = x.shape
        device = x.device

        embeds = self.dropout(self.embedding(x))  # (B, T, E)

        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers, batch, self.hidden_size, device=device
            )

        # h[layer] = (batch, hidden_size)
        h = [h_0[layer] for layer in range(self.num_layers)]

        all_output = []
        for t in range(seq_len):
            inp = embeds[:, t, :]  # (B, E) at time t
            for layer_idx, cell in enumerate(self.cells):
                h_new = cell(inp, h[layer_idx])
                inp = self.dropout(h_new) if layer_idx < self.num_layers - 1 else h_new
                h[layer_idx] = h_new
            all_output.append(inp)  # output of top layer at time t

        # Stack time outputs: (B, T, H)
        output = torch.stack(all_output, dim=1)
        logits = self.output_proj(output)          # (B, T, vocab_size)
        h_n = torch.stack(h, dim=0)               # (num_layers, B, H)
        return logits, h_n

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero-initialised hidden states."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
