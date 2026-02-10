"""
Export model weights and intermediate activations for C++ numerical parity testing.

Creates a test data directory containing:
  - model.grasslm       : exported model weights
  - input_ids.bin        : input token IDs (uint32 length + int32 array)
  - embed_output.bin     : h after tok+pos embeddings (tensor binary)
  - block_N_output.bin   : h after each Grassmann block (tensor binary)
  - final_norm_output.bin: h after final layer norm (tensor binary)
  - logits.bin           : final logits (tensor binary)

Tensor binary format:
  uint32 ndim
  uint32 dim[0], dim[1], ...
  float32 data[numel]  (row-major, C-contiguous)

Usage:
    python export_test_data.py --checkpoint checkpoints/best_model.pt --output test_data/
    python export_test_data.py --output test_data/  # uses a fresh random model
"""

import argparse
import os
import struct

import numpy as np
import torch

from grasslm.export import export_model
from grasslm.model import GrassLM


def write_tensor(path: str, tensor: torch.Tensor) -> None:
    """Write a tensor to binary format: ndim, shape[], float32 data[]."""
    t = tensor.cpu().detach().float().contiguous()
    with open(path, "wb") as f:
        shape = list(t.shape)
        f.write(struct.pack("<I", len(shape)))
        for dim in shape:
            f.write(struct.pack("<I", dim))
        f.write(t.numpy().tobytes())


def write_token_ids(path: str, token_ids: list[int]) -> None:
    """Write token IDs to binary: uint32 length, int32 ids[]."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(token_ids)))
        for tid in token_ids:
            f.write(struct.pack("<i", tid))


def export_test_data(
    output_dir: str,
    checkpoint_path: str | None = None,
    seq_len: int = 16,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)

    # Load or create model
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = ckpt["args"]
        model = GrassLM(
            vocab_size=saved_args.get("vocab_size", 30522),
            d_model=saved_args["d_model"],
            n_layers=saved_args["n_layers"],
            d_reduce=saved_args["d_reduce"],
            d_ff=saved_args["d_ff"],
            max_seq_len=saved_args["seq_len"],
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        model = GrassLM(
            vocab_size=64,
            d_model=16,
            n_layers=2,
            d_reduce=4,
            d_ff=32,
            max_seq_len=32,
            dropout=0.0,
            window_schedule=[1, 2],
        )
        print("Created fresh model (small config for testing)")

    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Export model weights
    model_path = os.path.join(output_dir, "model.grasslm")
    export_model(model, model_path)
    print(f"Exported model to {model_path}")

    # Generate deterministic input
    torch.manual_seed(seed)
    vocab_size = model.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    token_list = input_ids[0].tolist()

    write_token_ids(os.path.join(output_dir, "input_ids.bin"), token_list)
    print(f"Input IDs ({len(token_list)} tokens): {token_list}")

    # Forward pass with hooks to capture intermediate activations
    intermediates = {}

    # Capture embedding output (before dropout, which is disabled at eval)
    def hook_embed(module, input, output):
        # This captures the positional embedding lookup — but we need the sum.
        # Instead, we'll compute it manually below.
        pass

    # Run the forward pass step by step to capture intermediates
    with torch.no_grad():
        B, L = input_ids.shape
        positions = torch.arange(L).unsqueeze(0)
        h = model.tok_embed(input_ids) + model.pos_embed(positions)
        # embed_dropout is identity in eval mode with dropout=0.0
        h = model.embed_dropout(h)
        intermediates["embed_output"] = h.clone()

        for i, block in enumerate(model.blocks):
            h = block(h)
            intermediates[f"block_{i}_output"] = h.clone()

        h = model.ln_final(h)
        intermediates["final_norm_output"] = h.clone()

        logits = model.lm_head(h)
        intermediates["logits"] = logits.clone()

    # Write all intermediates (squeeze batch dim since C++ uses unbatched)
    for name, tensor in intermediates.items():
        # Remove batch dimension: (1, L, D) -> (L, D)
        t = tensor.squeeze(0)
        path = os.path.join(output_dir, f"{name}.bin")
        write_tensor(path, t)
        print(f"  {name}: shape {list(t.shape)}")

    print(f"\nTest data exported to {output_dir}/")
    print(f"Files: model.grasslm, input_ids.bin, "
          f"{len(intermediates)} activation files")


def main():
    parser = argparse.ArgumentParser(
        description="Export test data for C++ numerical parity tests"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). If omitted, uses a small random model.",
    )
    parser.add_argument(
        "--output", type=str, default="test_data",
        help="Output directory for test data (default: test_data)",
    )
    parser.add_argument(
        "--seq_len", type=int, default=16,
        help="Sequence length for test input (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    export_test_data(args.output, args.checkpoint, args.seq_len, args.seed)


if __name__ == "__main__":
    main()
