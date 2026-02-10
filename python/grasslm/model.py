"""
GrassLM: Attention-free causal language model based on Grassmann flows.

Implements the full causal language model from https://arxiv.org/abs/2512.19428, composing
token/positional embeddings, N stacked GrassmannBlocks with a multi-scale
window schedule, and a weight-tied LM head.
"""

import torch
import torch.nn as nn

from grasslm.layers import GrassmannBlock

# Default window schedule: each layer gets one offset
WINDOW_SCHEDULES = {
    6: [1, 2, 4, 8, 12, 16],
    12: [1, 1, 2, 2, 4, 4, 8, 8, 12, 12, 16, 16],
}


class GrassLM(nn.Module):
    """
    Causal Grassmann Language Model.

    Architecture:
        input_ids → tok_embed + pos_embed
        → N x GrassmannBlock (each with its own window offset)
        → LayerNorm
        → LM head (weight-tied with tok_embed)
        → logits

    Args:
        vocab_size: Vocabulary size (default: 30522 for BERT WordPiece)
        d_model: Model/hidden dimension
        n_layers: Number of GrassmannBlock layers
        d_reduce: Reduced dimension for Plücker encoding
        d_ff: Feed-forward intermediate dimension
        max_seq_len: Maximum sequence length for positional embeddings
        dropout: Dropout probability
        window_schedule: Per-layer window offsets. If None, uses built-in
            schedule for n_layers=6 or 12, otherwise defaults to [1] per layer.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_layers: int = 6,
        d_reduce: int = 32,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        window_schedule: list[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_reduce = d_reduce
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Determine window schedule
        if window_schedule is not None:
            schedule = window_schedule
        elif n_layers in WINDOW_SCHEDULES:
            schedule = WINDOW_SCHEDULES[n_layers]
        else:
            schedule = [1] * n_layers

        if len(schedule) != n_layers:
            raise ValueError(
                f"Window schedule length ({len(schedule)}) must match "
                f"n_layers ({n_layers})"
            )

        # Stacked Grassmann blocks — each gets one offset from the schedule
        self.blocks = nn.ModuleList([
            GrassmannBlock(
                d_model=d_model,
                d_reduce=d_reduce,
                d_ff=d_ff,
                window_offsets=[schedule[i]],
                dropout=dropout,
            )
            for i in range(n_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)

        # LM head — weight-tied with token embeddings
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        #Initialize weights following standard transformer conventions.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input token IDs to logits.

        Args:
            input_ids: Token indices, shape (B, L) with values in [0, vocab_size)

        Returns:
            logits: Unnormalized log-probs, shape (B, L, vocab_size)
        """
        B, L = input_ids.shape
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}"
            )

        # Token + positional embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        h = self.tok_embed(input_ids) + self.pos_embed(positions)
        h = self.embed_dropout(h)

        # Pass through Grassmann blocks
        for block in self.blocks:
            h = block(h)

        # Final layer norm + LM head
        h = self.ln_final(h)
        logits = self.lm_head(h)  # (B, L, vocab_size)

        return logits
