"""
Tests for the full GrassLM model.

Covers:
- Forward pass output shapes
- Gradient flow through all parameters
- Weight tying (lm_head shares tok_embed weights)
- Window schedule validation
- Sequence length bounds
"""

import torch
import pytest

from grasslm.model import GrassLM, WINDOW_SCHEDULES

# Small config for fast tests
VOCAB = 64
D_MODEL = 16
N_LAYERS = 2
D_REDUCE = 4
D_FF = 32
MAX_SEQ = 32
B = 2


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


class TestForwardPass:
    def test_output_shape(self, model):
        ids = torch.randint(0, VOCAB, (B, 16))
        logits = model(ids)
        assert logits.shape == (B, 16, VOCAB)

    def test_output_shape_single_token(self, model):
        ids = torch.randint(0, VOCAB, (1, 1))
        logits = model(ids)
        assert logits.shape == (1, 1, VOCAB)

    def test_output_shape_max_seq_len(self, model):
        ids = torch.randint(0, VOCAB, (1, MAX_SEQ))
        logits = model(ids)
        assert logits.shape == (1, MAX_SEQ, VOCAB)

    def test_exceeds_max_seq_len_raises(self, model):
        ids = torch.randint(0, VOCAB, (1, MAX_SEQ + 1))
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(ids)

    def test_output_dtype(self, model):
        ids = torch.randint(0, VOCAB, (B, 10))
        logits = model(ids)
        assert logits.dtype == torch.float32

    def test_batch_independence(self, model):
        # Each sample in a batch should be processed independently
        model.eval()
        ids = torch.randint(0, VOCAB, (3, 10))

        full_logits = model(ids)
        for i in range(3):
            single_logits = model(ids[i : i + 1])
            assert torch.allclose(full_logits[i], single_logits[0], atol=1e-5)

    def test_deterministic_eval(self, model):
        model.eval()
        ids = torch.randint(0, VOCAB, (B, 10))
        out1 = model(ids)
        out2 = model(ids)
        assert torch.allclose(out1, out2)


class TestGradientFlow:
    def test_all_parameters_get_gradients(self, model):
        ids = torch.randint(0, VOCAB, (B, 10))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            # At least some gradients should be nonzero
            # (weight-tied params share grad, that's fine)

    def test_loss_backward(self, model):
        ids = torch.randint(0, VOCAB, (B, 10))
        labels = torch.randint(0, VOCAB, (B, 10))

        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB), labels.view(-1)
        )
        loss.backward()

        # Verify loss is a finite scalar
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_magnitude_reasonable(self, model):
        ids = torch.randint(0, VOCAB, (B, 10))
        logits = model(ids)
        loss = logits.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )


class TestWeightTying:
    def test_lm_head_shares_embedding_weight(self, model):
        assert model.lm_head.weight is model.tok_embed.weight

    def test_lm_head_has_no_bias(self, model):
        assert model.lm_head.bias is None

    def test_shared_weight_gradient(self, model):
        ids = torch.randint(0, VOCAB, (B, 10))
        logits = model(ids)
        logits.sum().backward()

        # The shared weight should have a gradient
        assert model.tok_embed.weight.grad is not None
        # lm_head.weight IS tok_embed.weight, so same object
        assert model.lm_head.weight.grad is model.tok_embed.weight.grad


class TestWindowSchedule:
    def test_default_6_layer_schedule(self):
        m = GrassLM(
            vocab_size=VOCAB, d_model=D_MODEL, n_layers=6,
            d_reduce=D_REDUCE, d_ff=D_FF, max_seq_len=MAX_SEQ, dropout=0.0,
        )
        expected = WINDOW_SCHEDULES[6]
        for i, blk in enumerate(m.blocks):
            assert blk.mixing.window_offsets == [expected[i]]

    def test_default_12_layer_schedule(self):
        m = GrassLM(
            vocab_size=VOCAB, d_model=D_MODEL, n_layers=12,
            d_reduce=D_REDUCE, d_ff=D_FF, max_seq_len=MAX_SEQ, dropout=0.0,
        )
        expected = WINDOW_SCHEDULES[12]
        for i, blk in enumerate(m.blocks):
            assert blk.mixing.window_offsets == [expected[i]]

    def test_custom_schedule(self):
        m = GrassLM(
            vocab_size=VOCAB, d_model=D_MODEL, n_layers=3,
            d_reduce=D_REDUCE, d_ff=D_FF, max_seq_len=MAX_SEQ, dropout=0.0,
            window_schedule=[1, 3, 7],
        )
        for i, offset in enumerate([1, 3, 7]):
            assert m.blocks[i].mixing.window_offsets == [offset]

    def test_mismatched_schedule_raises(self):
        with pytest.raises(ValueError, match="must match"):
            GrassLM(
                vocab_size=VOCAB, d_model=D_MODEL, n_layers=3,
                d_reduce=D_REDUCE, d_ff=D_FF, max_seq_len=MAX_SEQ,
                window_schedule=[1, 2],  # only 2 offsets for 3 layers
            )

    def test_fallback_schedule_for_unknown_n_layers(self):
        m = GrassLM(
            vocab_size=VOCAB, d_model=D_MODEL, n_layers=5,
            d_reduce=D_REDUCE, d_ff=D_FF, max_seq_len=MAX_SEQ, dropout=0.0,
        )
        # Should default to [1] per layer
        for blk in m.blocks:
            assert blk.mixing.window_offsets == [1]


class TestModelConfig:
    def test_param_count_positive(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_n_blocks(self, model):
        assert len(model.blocks) == N_LAYERS

    def test_attributes_stored(self, model):
        assert model.vocab_size == VOCAB
        assert model.d_model == D_MODEL
        assert model.n_layers == N_LAYERS
        assert model.d_reduce == D_REDUCE
        assert model.d_ff == D_FF
        assert model.max_seq_len == MAX_SEQ
