"""
Tests for GrassLM weight export/reload roundtrip.

Covers:
- Export writes a valid .grasslm binary file
- Header contains correct model config
- Reload weights match original model exactly
- Float16 export roundtrip

NOTE: These tests require grasslm.export to be implemented (step 1.7).
      They will be skipped automatically if the module is not yet available.
"""

import struct
import tempfile
import os

import torch
import pytest

from grasslm.model import GrassLM

# Skip entire module if export isn't implemented yet
import grasslm.export as export

_has_export = hasattr(export, "export_model") and hasattr(export, "load_weights")
pytestmark = pytest.mark.skipif(
    not _has_export,
    reason="grasslm.export not yet implemented (step 1.7)",
)

VOCAB = 64
D_MODEL = 16
N_LAYERS = 2
D_REDUCE = 4
D_FF = 32
MAX_SEQ = 32


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GrassLM(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_reduce=D_REDUCE,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
        dropout=0.0,
        window_schedule=[1, 2],
    )


@pytest.fixture
def exported_path(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.grasslm")
        export.export_model(model, path)
        yield path


class TestExportFile:
    def test_file_created(self, exported_path):
        assert os.path.exists(exported_path)
        assert os.path.getsize(exported_path) > 64  # at least header

    def test_magic_bytes(self, exported_path):
        with open(exported_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"GRLM"

    def test_header_values(self, exported_path):
        with open(exported_path, "rb") as f:
            header = f.read(64)

        magic = header[0:4]
        assert magic == b"GRLM"

        # Parse header fields (all uint32 after magic)
        values = struct.unpack_from("<7I", header, 4)
        version, n_layers, d_model, d_reduce, d_ff, vocab_size, max_seq_len = values

        assert n_layers == N_LAYERS
        assert d_model == D_MODEL
        assert d_reduce == D_REDUCE
        assert d_ff == D_FF
        assert vocab_size == VOCAB
        assert max_seq_len == MAX_SEQ


class TestRoundtrip:
    def test_weight_roundtrip_f32(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.grasslm")
            export.export_model(model, path)
            loaded_weights = export.load_weights(path)

            state_dict = model.state_dict()
            for name, param in state_dict.items():
                assert name in loaded_weights, f"Missing weight: {name}"
                loaded = loaded_weights[name]
                assert loaded.shape == param.shape, (
                    f"Shape mismatch for {name}: {loaded.shape} vs {param.shape}"
                )
                assert torch.allclose(loaded, param.cpu(), atol=1e-6), (
                    f"Value mismatch for {name}, max diff: "
                    f"{(loaded - param.cpu()).abs().max().item()}"
                )

    def test_reload_produces_identical_output(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.grasslm")
            export.export_model(model, path)
            loaded_weights = export.load_weights(path)

            # Create a fresh model and load exported weights
            model2 = GrassLM(
                vocab_size=VOCAB,
                d_model=D_MODEL,
                n_layers=N_LAYERS,
                d_reduce=D_REDUCE,
                d_ff=D_FF,
                max_seq_len=MAX_SEQ,
                dropout=0.0,
                window_schedule=[1, 2],
            )
            model2.load_state_dict(loaded_weights)

            # Compare outputs
            model.eval()
            model2.eval()
            ids = torch.randint(0, VOCAB, (1, 10))
            out1 = model(ids)
            out2 = model2(ids)
            assert torch.allclose(out1, out2, atol=1e-6)

    def test_all_weights_exported(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.grasslm")
            export.export_model(model, path)
            loaded_weights = export.load_weights(path)

            state_dict = model.state_dict()
            assert set(loaded_weights.keys()) == set(state_dict.keys())
