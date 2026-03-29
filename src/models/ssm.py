"""
S4 (Structured State Space Sequence Model) implemented in PyTorch.

Based on:
    Gu, Goel & Ré (2022) "Efficiently Modeling Long Sequences with
    Structured State Spaces" (ICLR 2022)
and the companion annotated blog post by Rush & Karamcheti.

Key components:
    make_HiPPO(N)        – HiPPO transition matrix
    make_NPLR_HiPPO(N)   – Normal Plus Low-Rank form
    make_DPLR_HiPPO(N)   – Diagonal Plus Low-Rank via eigen-decomposition
    kernel_DPLR(...)      – compute SSM convolution kernel via Cauchy
    discrete_DPLR(...)    – discretise SSM to RNN for inference
    S4Layer              – single S4 layer (CNN training / RNN inference)
    S4Model              – stacked S4 model for Next Word Prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# HiPPO initialisation
# ---------------------------------------------------------------------------

def make_HiPPO(N: int) -> torch.Tensor:
    """
    Construct the NxN HiPPO matrix (negated, as used in the S4 paper).

    A_nk = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
          -(n+1)                          if n = k
           0                              if n < k
    """
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))   # (N,)
    A = P.unsqueeze(1) * P.unsqueeze(0)   # outer product
    A = torch.tril(A) - torch.diag(torch.arange(N, dtype=torch.float32))
    return -A


def make_NPLR_HiPPO(N: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return HiPPO matrix alongside its NPLR low-rank correction and B vector.

    Returns:
        A: (N, N) HiPPO matrix (neg)
        P: (N,)   low-rank correction vector
        B: (N,)   input projection vector
    """
    A = make_HiPPO(N)
    P = torch.sqrt(0.5 + torch.arange(N, dtype=torch.float32))     # (N,)
    B = torch.sqrt(2 * torch.arange(N, dtype=torch.float32) + 1.0) # (N,)
    return A, P, B


def make_DPLR_HiPPO(
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Diagonalise the NPLR HiPPO matrix to obtain DPLR parameters.

    Returns:
        Lambda: (N,) complex diagonal eigenvalues
        P:      (N,) complex low-rank vector (in diagonal basis)
        B:      (N,) complex input vector   (in diagonal basis)
        V:      (N, N) complex unitary eigenvector matrix
    """
    A, P_real, B_real = make_NPLR_HiPPO(N)

    # Symmetrised normal part: S = A + P P^T  (skew-symmetric + diag)
    S = A + P_real.unsqueeze(1) * P_real.unsqueeze(0)   # (N, N)

    # Real part of eigenvalues: mean of diagonal of S (all equal for HiPPO)
    S_diag = torch.diagonal(S)
    Lambda_real = torch.full(S_diag.shape, S_diag.mean())

    # Diagonalise -iS (Hermitian) to get imaginary part of Lambda
    # eigh returns ascending eigenvalues
    Lambda_imag, V = torch.linalg.eigh(-1j * S.to(torch.complex64))

    # Project P and B into the diagonal basis
    P_c = (V.conj().T @ P_real.to(torch.complex64))   # (N,)
    B_c = (V.conj().T @ B_real.to(torch.complex64))   # (N,)

    Lambda = Lambda_real.to(torch.complex64) + 1j * Lambda_imag
    return Lambda, P_c, B_c, V


# ---------------------------------------------------------------------------
# Cauchy kernel
# ---------------------------------------------------------------------------

def cauchy(v: torch.Tensor, omega: torch.Tensor, lambd: torch.Tensor) -> torch.Tensor:
    """
    Compute the Cauchy matrix–vector product.

    Calculates sum_n v_n / (omega_l - lambda_n) for each l.

    Args:
        v:      (N,) complex
        omega:  (L,) complex
        lambd:  (N,) complex

    Returns:
        result: (L,) complex
    """
    # Broadcasting: (L, 1) - (1, N) → (L, N)
    diffs = omega.unsqueeze(1) - lambd.unsqueeze(0)   # (L, N)
    return (v.unsqueeze(0) / diffs).sum(dim=1)         # (L,)


# ---------------------------------------------------------------------------
# Kernel and discretisation
# ---------------------------------------------------------------------------

def kernel_DPLR(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    step: float,
    L: int,
) -> torch.Tensor:
    """
    Compute the length-L SSM convolution kernel for a DPLR matrix via the
    Cauchy kernel trick (fast, O(N*L) instead of O(N^2*L)).

    All inputs are complex (or will be cast to complex).

    Args:
        Lambda: (N,) diagonal eigenvalues
        P:      (N,) low-rank vector (left)
        Q:      (N,) low-rank vector (right, usually == P)
        B:      (N,) input projection
        C:      (N,) output projection  (already absorbs Cbar correction)
        step:   discretisation step Δ
        L:      sequence length

    Returns:
        kernel: (L,) real-valued convolution filter
    """
    # Use float32 throughout; keep on CPU — complex tensors not supported on MPS/CUDA
    Omega_L = torch.exp(
        torch.arange(L, dtype=torch.float32) * (-2j * math.pi / L)
    ).to(torch.complex64)

    # g = (2/Δ) * (1 - z) / (1 + z)  (bilinear transform)
    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c_scale = 2.0 / (1.0 + Omega_L)  # overall scalar

    # Compute four Cauchy products for Woodbury identity
    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)

    at_roots = c_scale * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    # Inverse FFT to recover filter
    out = torch.fft.ifft(at_roots, n=L)
    return out.real.to(torch.float32)


def discrete_DPLR(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    step: float,
    L: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the discrete-time SSM matrices (Abar, Bbar, Cbar) for RNN mode.

    Uses the bilinear (Tustin) method with Woodbury identity for efficiency.

    Returns:
        Ab: (N, N) complex
        Bb: (N, 1) complex
        Cb: (1, N) complex
    """
    N = Lambda.shape[0]
    B = B.unsqueeze(1)        # (N, 1)
    C = C.unsqueeze(0)        # (1, N)

    A = torch.diag(Lambda) - P.unsqueeze(1) @ Q.unsqueeze(0).conj()  # (N,N)
    I = torch.eye(N, dtype=torch.complex64, device=Lambda.device)

    # Forward Euler: A0 = (2/Δ)*I + A
    A0 = (2.0 / step) * I + A

    # Backward Euler via Woodbury on (2/Δ - Λ)
    D = torch.diag(1.0 / ((2.0 / step) - Lambda))   # (N, N) diagonal
    P2 = P.unsqueeze(1)                              # (N, 1)
    Qc = Q.conj().unsqueeze(0)                       # (1, N)
    A1 = D - D @ P2 * (1.0 / (1.0 + (Qc @ D @ P2))) * (Qc @ D)  # (N,N)

    Ab = A1 @ A0
    Bb = 2.0 * A1 @ B

    # Recover Cbar: C (I - Ab^L)^{-1}   (conjugated to match generating fn)
    Ab_L = torch.linalg.matrix_power(Ab, L)
    Cb = C @ torch.linalg.inv(I - Ab_L).conj()
    return Ab, Bb, Cb.conj()


# ---------------------------------------------------------------------------
# S4 Layer
# ---------------------------------------------------------------------------

class S4Layer(nn.Module):
    """
    A single S4 layer.

    During training (decode=False) uses the efficient CNN (convolution) path.
    During inference (decode=True) uses the RNN recurrence path with cached state.

    H independent SSMs are vmapped (here implemented as a batched loop,
    stored as a single parameter with an extra leading dimension of size H).

    Args:
        d_model:  number of independent SSMs (feature channels)
        N:        SSM state dimension
        l_max:    maximum sequence length (needed to pre-compute kernel)
        dropout:  dropout on the output
        decode:   True for RNN (inference) mode
    """

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        l_max: int = 1024,
        dropout: float = 0.0,
        decode: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.l_max = l_max
        self.decode = decode

        # Initialise DPLR HiPPO parameters once, then store as learnable
        Lambda, P, B, _ = make_DPLR_HiPPO(N)

        # Replicate across d_model channels and make learnable
        # Real and imaginary parts stored separately for stability
        self.Lambda_re = nn.Parameter(
            Lambda.real.unsqueeze(0).expand(d_model, -1).clone()  # (H, N)
        )
        self.Lambda_im = nn.Parameter(
            Lambda.imag.unsqueeze(0).expand(d_model, -1).clone()  # (H, N)
        )
        self.P = nn.Parameter(
            torch.stack([P.real, P.imag], dim=-1)
            .unsqueeze(0).expand(d_model, -1, -1).clone()  # (H, N, 2)
        )
        self.B = nn.Parameter(
            torch.stack([B.real, B.imag], dim=-1)
            .unsqueeze(0).expand(d_model, -1, -1).clone()  # (H, N, 2)
        )
        # C initialised as standard normal (stored as real + imag pairs)
        self.C = nn.Parameter(torch.randn(d_model, N, 2) * (0.5 ** 0.5))  # (H,N,2)
        self.D = nn.Parameter(torch.ones(d_model))                         # (H,)
        self.log_step = nn.Parameter(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )  # (H,)

        self.dropout = nn.Dropout(dropout)

        # RNN-mode cache
        if decode:
            # Register as buffers (not learnable, but part of state)
            self.register_buffer(
                "cache_x",
                torch.zeros(d_model, N, dtype=torch.complex64),
            )

    def _get_params(self) -> Tuple[torch.Tensor, ...]:
        """Reconstruct complex parameters from stored real/imag pairs."""
        Lambda = torch.clamp(self.Lambda_re, max=-1e-4) + 1j * self.Lambda_im
        P = self.P[..., 0] + 1j * self.P[..., 1]
        B = self.B[..., 0] + 1j * self.B[..., 1]
        C = self.C[..., 0] + 1j * self.C[..., 1]
        step = torch.exp(self.log_step)   # (H,)
        return Lambda, P, B, C, step

    def _compute_kernel(self) -> torch.Tensor:
        """Compute the SSM convolution kernel for all H channels. Returns (H, L).

        The kernel is computed on CPU (complex64 is not supported on MPS/CUDA)
        and then moved to the model's device.
        """
        Lambda, P, B, C, step = self._get_params()
        # Move all complex-valued inputs to CPU for kernel computation
        Lambda_cpu = Lambda.cpu()
        P_cpu = P.cpu()
        B_cpu = B.cpu()
        C_cpu = C.cpu()
        target_device = Lambda.device
        kernels = []
        for h in range(self.d_model):
            k = kernel_DPLR(
                Lambda_cpu[h], P_cpu[h], P_cpu[h], B_cpu[h], C_cpu[h],
                step=step[h].item(), L=self.l_max,
            )
            kernels.append(k)
        return torch.stack(kernels, dim=0).to(target_device)  # (H, L)

    def _cnn_forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Convolutional path (training).

        Args:
            u: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        B_sz, L, H = u.shape
        K = self._compute_kernel()  # (H, L_max)
        K = K[:, :L]                # (H, L)

        # Convolve each channel independently via FFT
        # u:  (B, L, H) → transpose to (H, B, L) for easy FFT
        u_t = u.permute(2, 0, 1)   # (H, B, L)

        fft_size = 2 * L
        U = torch.fft.rfft(u_t, n=fft_size)            # (H, B, fft_size//2+1)
        K_f = torch.fft.rfft(K, n=fft_size).unsqueeze(1)  # (H, 1, fft_size//2+1)
        Y = torch.fft.irfft(U * K_f, n=fft_size)[..., :L]  # (H, B, L)

        y = Y.permute(1, 2, 0)  # (B, L, H)
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)  # skip connection
        return y

    def _rnn_forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Recurrent path (inference, one step at a time).

        Args:
            u: (batch=1, seq_len, d_model)  [batch must be 1 for cached state]

        Returns:
            y: (batch=1, seq_len, d_model)
        """
        Lambda, P, B_param, C, step = self._get_params()
        outputs = []

        for h in range(self.d_model):
            Ab, Bb, Cb = discrete_DPLR(
                Lambda[h], P[h], P[h], B_param[h], C[h],
                step=step[h].item(), L=self.l_max,
            )
            x = self.cache_x[h].clone()   # (N,) complex
            h_outs = []
            for t in range(u.shape[1]):
                u_t = u[0, t, h].to(torch.complex64)
                x = Ab @ x + Bb.squeeze(1) * u_t
                y_t = (Cb @ x).real + self.D[h] * u[0, t, h]
                h_outs.append(y_t)
            self.cache_x[h] = x.detach()
            outputs.append(torch.stack(h_outs))  # (L,)

        y = torch.stack(outputs, dim=1).unsqueeze(0)  # (1, L, H)
        return y

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        if self.decode:
            return self._rnn_forward(u)
        return self.dropout(self._cnn_forward(u))


# ---------------------------------------------------------------------------
# S4 Model (stacked, for Next Word Prediction)
# ---------------------------------------------------------------------------

class SequenceBlock(nn.Module):
    """
    One S4Layer wrapped with LayerNorm, GELU, GLU gating, and a residual.

    Follows the SequenceBlock design from the Annotated S4.
    """

    def __init__(
        self,
        d_model: int,
        N: int,
        l_max: int,
        dropout: float = 0.1,
        decode: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model=d_model, N=N, l_max=l_max,
                          dropout=dropout, decode=decode)
        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, d_model)   # GLU gate
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.s4(x)
        x = self.drop(F.gelu(x))
        # Gated Linear Unit (GLU)
        x = self.out1(x) * torch.sigmoid(self.out2(x))
        x = residual + self.drop(x)
        return x


class S4Model(nn.Module):
    """
    Stacked S4 model for Next Word Prediction (NWP).

    Embedding → n_layers × SequenceBlock → Linear → log-softmax

    Args:
        vocab_size:  word vocabulary size
        d_model:     embedding / hidden dimension
        N:           SSM state size per channel
        n_layers:    number of stacked S4 blocks
        l_max:       maximum sequence length
        dropout:     dropout probability
        pad_idx:     padding index
        decode:      True at inference (use RNN path)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        N: int = 64,
        n_layers: int = 4,
        l_max: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
        decode: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.blocks = nn.ModuleList(
            [
                SequenceBlock(d_model=d_model, N=N, l_max=l_max,
                              dropout=dropout, decode=decode)
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer word indices

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        emb = self.dropout(self.embedding(x))   # (B, T, d_model)
        h = emb
        for block in self.blocks:
            h = block(h)
        logits = self.output_proj(h)            # (B, T, vocab_size)
        return logits
