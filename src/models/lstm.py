"""
LSTM implemented from scratch using only basic PyTorch operations.
nn.RNN and nn.LSTM are NOT used.

Based on Hochreiter & Schmidhuber (1997) "Long Short-Term Memory".

Gate equations (concatenated-input form for efficiency):
    f_t = σ(W_f [h_{t-1}, x_t] + b_f)   forget gate
    i_t = σ(W_i [h_{t-1}, x_t] + b_i)   input gate
    g_t = tanh(W_g [h_{t-1}, x_t] + b_g) cell gate (candidate)
    o_t = σ(W_o [h_{t-1}, x_t] + b_o)   output gate
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t    cell state (CEC)
    h_t = o_t ⊙ tanh(c_t)               hidden state
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMCell(nn.Module):
    """Single-step LSTM cell.

    Args:
        input_size:  dimensionality of input vector
        hidden_size: dimensionality of hidden / cell state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # All four gates computed in a single matrix multiply for efficiency.
        # W maps [x_t, h_{t-1}] → [f, i, g, o] concatenated (4 * hidden_size)
        self.W = nn.Parameter(torch.empty(4 * hidden_size, input_size + hidden_size))
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise weights with Kaiming uniform; bias initialised to zero
        except the forget gate bias which is set to 1.0 to encourage remembering
        at initialisation (Gers et al., 2000)."""
        nn.init.kaiming_uniform_(self.W, nonlinearity="sigmoid")
        # forget gate slice: indices [0 : hidden_size]
        with torch.no_grad():
            self.b[: self.hidden_size].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:     (batch, input_size)
            state: tuple of (h_prev, c_prev), each (batch, hidden_size)

        Returns:
            (h_t, c_t): each (batch, hidden_size)
        """
        h_prev, c_prev = state
        H = self.hidden_size

        # Concatenate input and previous hidden state along feature dim
        combined = torch.cat([x, h_prev], dim=1)          # (B, input+hidden)
        gates = combined @ self.W.t() + self.b             # (B, 4*H)

        f = torch.sigmoid(gates[:, :H])                    # forget gate
        i = torch.sigmoid(gates[:, H : 2 * H])            # input gate
        g = torch.tanh(gates[:, 2 * H : 3 * H])           # cell gate
        o = torch.sigmoid(gates[:, 3 * H :])               # output gate

        c_t = f * c_prev + i * g                           # CEC update
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class LSTM(nn.Module):
    """
    Multi-layer stacked LSTM for sequence-to-sequence character-level decryption.

    Args:
        vocab_size:        size of the *output* (plain) character vocabulary
        embed_dim:         embedding dimensionality
        hidden_size:       LSTM hidden/cell state size
        num_layers:        number of stacked LSTM layers
        dropout:           dropout probability between layers
        pad_idx:           padding index for the *output* vocab
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

        # One LSTMCell per layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = embed_dim if layer == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        # Output projection: plain character vocab
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:     (batch, seq_len) integer token indices
            state: optional tuple (h_0, c_0), each (num_layers, batch, hidden_size)

        Returns:
            logits: (batch, seq_len, vocab_size)
            (h_n, c_n): each (num_layers, batch, hidden_size)
        """
        batch, seq_len = x.shape
        device = x.device

        embeds = self.dropout(self.embedding(x))  # (B, T, E)

        if state is None:
            zeros = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
            state = (zeros, zeros.clone())

        h = [state[0][layer] for layer in range(self.num_layers)]
        c = [state[1][layer] for layer in range(self.num_layers)]

        all_output = []
        for t in range(seq_len):
            inp = embeds[:, t, :]  # (B, E)
            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(inp, (h[layer_idx], c[layer_idx]))
                inp = self.dropout(h_new) if layer_idx < self.num_layers - 1 else h_new
                h[layer_idx] = h_new
                c[layer_idx] = c_new
            all_output.append(inp)  # output of top layer at time t

        output = torch.stack(all_output, dim=1)   # (B, T, H)
        logits = self.output_proj(output)          # (B, T, vocab_size)
        h_n = torch.stack(h, dim=0)               # (num_layers, B, H)
        c_n = torch.stack(c, dim=0)               # (num_layers, B, H)
        return logits, (h_n, c_n)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised (h, c) state tuple."""
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()
