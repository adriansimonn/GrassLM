"""
Core Grassmann layers for GrassLM.

Implements the Causal Grassmann mixing architecture from https://arxiv.org/abs/2512.19428:
- PluckerEncoder: reduces tokens to low-rank space, computes Plucker coordinates
- GrassmannMixing: multi-scale Plucker encoding with gated fusion
- GrassmannBlock: full block with mixing, layer norm, FFN, and residuals
"""

import torch
import torch.nn as nn


class PluckerEncoder(nn.Module):
    """
    Encodes local token pairs as points on Gr(2, r) via Plucker coordinates.

    For each position t and a given offset Δ, forms the pair (z_t, z_{t+Δ})
    in reduced space R^r, computes the Plucker embedding in R^{C(r,2)},
    normalizes, and projects back to model dimension.
    """

    def __init__(self, d_model: int = 256, d_reduce: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_reduce = d_reduce
        self.plucker_dim = d_reduce * (d_reduce - 1) // 2  # C(r, 2) = 496

        self.W_red = nn.Linear(d_model, d_reduce)
        self.W_plu = nn.Linear(self.plucker_dim, d_model)

        # Precompute index pairs (i, j) with i < j for Plucker coordinates
        idx = torch.combinations(torch.arange(d_reduce), r=2)  # (plucker_dim, 2)
        self.register_buffer("idx_i", idx[:, 0])  # (plucker_dim,)
        self.register_buffer("idx_j", idx[:, 1])  # (plucker_dim,)

    def forward(self, h: torch.Tensor, window_offset: int) -> torch.Tensor:
        """
        Compute Plucker features for token pairs at a given offset.

        Args:
            h: Hidden states, shape (B, L, d_model)
            window_offset: Causal offset Δ > 0; pairs position t with t-Δ

        Returns:
            g: Grassmann features, shape (B, L, d_model).
               First Δ positions (with no backward neighbor) are zero-padded.
        """
        B, L, _ = h.shape
        delta = window_offset

        # 1. Reduce to low-rank space
        z = self.W_red(h)  # (B, L, r)

        # 2. Form causal pairs: z_t paired with z_{t-delta} (look backward)
        if delta >= L:
            # No valid pairs exist
            return torch.zeros_like(h)

        z_t = z[:, delta:, :]           # (B, L-Δ, r)  — positions Δ..L-1
        z_td = z[:, :L - delta, :]      # (B, L-Δ, r)  — positions 0..L-Δ-1

        # 3. Compute Plucker coordinates: p[i,j] = z_t[i]*z_td[j] - z_t[j]*z_td[i]
        z_t_i = z_t[:, :, self.idx_i]   # (B, L-Δ, plucker_dim)
        z_t_j = z_t[:, :, self.idx_j]   # (B, L-Δ, plucker_dim)
        z_td_i = z_td[:, :, self.idx_i]  # (B, L-Δ, plucker_dim)
        z_td_j = z_td[:, :, self.idx_j]  # (B, L-Δ, plucker_dim)

        p = z_t_i * z_td_j - z_t_j * z_td_i  # (B, L-Δ, plucker_dim)

        # 4. Normalize
        p_hat = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # 5. Project back to model dimension
        g_valid = self.W_plu(p_hat)  # (B, L-Δ, d_model)

        # 6. Pad to full sequence length (first Δ positions have no backward neighbor)
        g = torch.zeros(B, L, self.d_model, device=h.device, dtype=h.dtype)
        g[:, delta:, :] = g_valid

        return g


class GrassmannMixing(nn.Module):
    """
    Multi-scale Grassmann mixing with gated fusion.

    For each offset in window_offsets, computes Plucker features via a
    shared PluckerEncoder, averages across valid offsets per position,
    and fuses with the original hidden states via a learned gate.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_reduce: int = 32,
        window_offsets: list[int] | None = None,
    ):
        super().__init__()
        if window_offsets is None:
            window_offsets = [1]
        self.window_offsets = window_offsets
        self.d_model = d_model

        # Shared encoder across offsets (as per paper: shared W_red)
        self.plucker_encoder = PluckerEncoder(d_model, d_reduce)

        # Gating: takes [h; g] and produces per-dimension gate α
        self.W_gate = nn.Linear(2 * d_model, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply Grassmann mixing.

        Args:
            h: Hidden states, shape (B, L, d_model)

        Returns:
            h_mix: Mixed hidden states, shape (B, L, d_model)
        """
        B, L, d = h.shape

        # 1. Compute Plucker features for each offset
        g_sum = torch.zeros(B, L, d, device=h.device, dtype=h.dtype)
        count = torch.zeros(B, L, 1, device=h.device, dtype=h.dtype)

        for delta in self.window_offsets:
            g_delta = self.plucker_encoder(h, delta)  # (B, L, d)
            g_sum = g_sum + g_delta

            # Track how many offsets are valid per position (positions delta..L-1)
            if delta < L:
                count[:, delta:, :] += 1.0

        # 2. Average over valid offsets
        g = g_sum / count.clamp(min=1.0)

        # 3. Gated fusion: α = sigmoid(W_gate([h; g]))
        u = torch.cat([h, g], dim=-1)  # (B, L, 2d)
        alpha = torch.sigmoid(self.W_gate(u))  # (B, L, d)

        # h_mix = α * h + (1 - α) * g
        h_mix = alpha * h + (1 - alpha) * g

        return h_mix


class GrassmannBlock(nn.Module):
    """
    A single Grassmann Transformer block.

    Replaces self-attention with Grassmann mixing, followed by
    layer norm, FFN with GELU, residual connections, and dropout.

    Architecture:
        h -> GrassmannMixing -> Dropout -> LayerNorm -> (+FFN -> Dropout) -> LayerNorm -> out
    with a residual connection around the FFN.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_reduce: int = 32,
        d_ff: int = 1024,
        window_offsets: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if window_offsets is None:
            window_offsets = [1]

        self.mixing = GrassmannMixing(d_model, d_reduce, window_offsets)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Grassmann block.

        Args:
            h: Input hidden states, shape (B, L, d_model)

        Returns:
            h_out: Output hidden states, shape (B, L, d_model)
        """
        # Grassmann mixing + dropout + layer norm
        h_mix = self.mixing(h)
        h_mix = self.dropout(self.ln1(h_mix))

        # FFN with residual + layer norm
        h_out = self.ln2(h_mix + self.dropout(self.ffn(h_mix)))

        return h_out
