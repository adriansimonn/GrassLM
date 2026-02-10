"""
Weight export/import for the .grasslm binary format.

Binary format (.grasslm):
    Header (64 bytes):
        magic: "GRLM" (4 bytes)
        version: uint32
        n_layers: uint32
        d_model: uint32
        d_reduce: uint32
        d_ff: uint32
        vocab_size: uint32
        max_seq_len: uint32
        dtype: uint32 (0=f32, 1=f16)
        padding: zeros to 64 bytes

    Weight table:
        num_weights: uint32
        For each weight:
            name_len: uint32
            name: bytes (UTF-8)
            n_dims: uint32
            shape: uint32[n_dims]
            data: float32[] or float16[] (row-major)
"""

import struct

import numpy as np
import torch

from grasslm.model import GrassLM

FORMAT_VERSION = 1

# Keys to skip when exporting (tied weights and precomputed buffers)
_SKIP_KEYS = {"lm_head.weight"}
_SKIP_SUBSTRINGS = {"idx_i", "idx_j"}


def _should_skip(key: str) -> bool:
    if key in _SKIP_KEYS:
        return True
    return any(sub in key for sub in _SKIP_SUBSTRINGS)


def export_model(model: GrassLM, path: str, dtype: str = "f32") -> None:
    """
    Export a GrassLM model to the .grasslm binary format.

    Args:
        model: Trained GrassLM model.
        path: Output file path.
        dtype: "f32" for float32 or "f16" for float16.
    """
    dtype_code = 0 if dtype == "f32" else 1

    state_dict = model.state_dict()
    weights = {k: v for k, v in state_dict.items() if not _should_skip(k)}

    # Add per-block window offsets as a metadata tensor so C++ uses the
    # exact same schedule that was used during training.
    window_offsets = [block.mixing.window_offsets[0] for block in model.blocks]
    weights["window_offsets"] = torch.tensor(window_offsets, dtype=torch.float32)

    with open(path, "wb") as f:
        # --- Header (64 bytes) ---
        header = bytearray(64)
        header[0:4] = b"GRLM"
        struct.pack_into("<I", header, 4, FORMAT_VERSION)
        struct.pack_into("<I", header, 8, model.n_layers)
        struct.pack_into("<I", header, 12, model.d_model)
        struct.pack_into("<I", header, 16, model.d_reduce)
        struct.pack_into("<I", header, 20, model.d_ff)
        struct.pack_into("<I", header, 24, model.vocab_size)
        struct.pack_into("<I", header, 28, model.max_seq_len)
        struct.pack_into("<I", header, 32, dtype_code)
        f.write(header)

        # --- Weight table ---
        f.write(struct.pack("<I", len(weights)))

        for name, param in weights.items():
            tensor = param.cpu().detach()

            # Name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Shape
            shape = list(tensor.shape)
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))

            # Data
            if dtype == "f16":
                data = tensor.to(torch.float16).numpy().tobytes()
            else:
                data = tensor.to(torch.float32).numpy().tobytes()
            f.write(data)


def load_weights(path: str) -> dict[str, torch.Tensor]:
    """
    Load weights from a .grasslm file back as a PyTorch state dict.

    The returned dict can be passed directly to model.load_state_dict().
    The lm_head.weight (tied with tok_embed.weight) is automatically included.

    Args:
        path: Path to the .grasslm file.

    Returns:
        Dictionary mapping weight names to tensors.
    """
    with open(path, "rb") as f:
        # --- Header ---
        header = f.read(64)
        magic = header[0:4]
        if magic != b"GRLM":
            raise ValueError(f"Invalid magic bytes: {magic!r}")

        n_layers = struct.unpack_from("<I", header, 8)[0]
        d_reduce = struct.unpack_from("<I", header, 16)[0]
        dtype_code = struct.unpack_from("<I", header, 32)[0]

        # --- Weight table ---
        (num_weights,) = struct.unpack("<I", f.read(4))

        weights = {}
        for _ in range(num_weights):
            # Name
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8")

            # Shape
            (n_dims,) = struct.unpack("<I", f.read(4))
            shape = [struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)]

            # Data
            numel = 1
            for d in shape:
                numel *= d

            if dtype_code == 1:
                raw = np.frombuffer(f.read(numel * 2), dtype=np.float16)
                tensor = torch.from_numpy(raw.astype(np.float32)).reshape(shape)
            else:
                raw = np.frombuffer(f.read(numel * 4), dtype=np.float32)
                tensor = torch.from_numpy(raw.copy()).reshape(shape)

            weights[name] = tensor

    # Remove metadata tensors that aren't part of the model state dict
    weights.pop("window_offsets", None)

    # Reconstruct the tied lm_head weight
    if "tok_embed.weight" in weights and "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["tok_embed.weight"]

    # Reconstruct precomputed index buffers (idx_i, idx_j) for each PluckerEncoder
    idx = torch.combinations(torch.arange(d_reduce), r=2)
    idx_i = idx[:, 0]
    idx_j = idx[:, 1]
    for i in range(n_layers):
        prefix = f"blocks.{i}.mixing.plucker_encoder."
        weights[prefix + "idx_i"] = idx_i
        weights[prefix + "idx_j"] = idx_j

    return weights
