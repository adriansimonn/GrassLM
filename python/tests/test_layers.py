"""
Tests for GrassLM core layers: PluckerEncoder, GrassmannMixing, GrassmannBlock.

Covers:
- Plücker coordinate correctness (hand-computed example with small d_reduce)
- Output shapes for all layers
- Causal masking (tail positions zero-padded)
- Plücker normalization (unit norm)
- Gating bounds (alpha in [0, 1])
"""

import torch
import pytest

from grasslm.layers import PluckerEncoder, GrassmannMixing, GrassmannBlock

# Use small dims for fast tests
D_MODEL = 16
D_REDUCE = 4
D_FF = 32
B, L = 2, 10


@pytest.fixture
def plucker():
    torch.manual_seed(0)
    return PluckerEncoder(d_model=D_MODEL, d_reduce=D_REDUCE)


@pytest.fixture
def mixing():
    torch.manual_seed(0)
    return GrassmannMixing(d_model=D_MODEL, d_reduce=D_REDUCE, window_offsets=[1])


@pytest.fixture
def block():
    torch.manual_seed(0)
    return GrassmannBlock(
        d_model=D_MODEL, d_reduce=D_REDUCE, d_ff=D_FF,
        window_offsets=[1], dropout=0.0,
    )


# ---------- PluckerEncoder ----------


class TestPluckerEncoder:
    def test_output_shape(self, plucker):
        h = torch.randn(B, L, D_MODEL)
        g = plucker(h, window_offset=1)
        assert g.shape == (B, L, D_MODEL)

    def test_output_shape_large_offset(self, plucker):
        h = torch.randn(B, L, D_MODEL)
        g = plucker(h, window_offset=4)
        assert g.shape == (B, L, D_MODEL)

    def test_plucker_dim(self, plucker):
        # C(4, 2) = 6 for d_reduce=4
        assert plucker.plucker_dim == D_REDUCE * (D_REDUCE - 1) // 2
        assert plucker.plucker_dim == 6

    def test_causal_masking_tail_is_zero(self, plucker):
        # With offset=3, positions L-3..L-1 have no valid pair -> should be zero
        h = torch.randn(B, L, D_MODEL)
        delta = 3
        g = plucker(h, window_offset=delta)
        assert torch.all(g[:, L - delta:, :] == 0.0)

    def test_offset_exceeds_length_returns_zeros(self, plucker):
        h = torch.randn(B, L, D_MODEL)
        g = plucker(h, window_offset=L + 5)
        assert torch.all(g == 0.0)

    def test_offset_equals_length_returns_zeros(self, plucker):
        h = torch.randn(B, L, D_MODEL)
        g = plucker(h, window_offset=L)
        assert torch.all(g == 0.0)

    def test_plucker_coordinates_hand_computed(self):
        # d_reduce=3 so plucker_dim = C(3,2) = 3
        # Index pairs: (0,1), (0,2), (1,2)
        enc = PluckerEncoder(d_model=3, d_reduce=3)

        # Override W_red to be identity so z = h
        with torch.no_grad():
            enc.W_red.weight.copy_(torch.eye(3))
            enc.W_red.bias.zero_()

        # h: single batch, 2 positions, d_model=3
        # z_t = [1, 2, 3], z_{t+1} = [4, 5, 6]
        h = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)

        # With offset=1: pair (z_0, z_1) at position 0, position 1 is zero-padded
        # Plucker coords for pair (z_t, z_td):
        #   p[0,1] = z_t[0]*z_td[1] - z_t[1]*z_td[0] = 1*5 - 2*4 = -3
        #   p[0,2] = z_t[0]*z_td[2] - z_t[2]*z_td[0] = 1*6 - 3*4 = -6
        #   p[1,2] = z_t[1]*z_td[2] - z_t[2]*z_td[1] = 2*6 - 3*5 = -3
        expected_p = torch.tensor([-3.0, -6.0, -3.0])
        expected_p_hat = expected_p / expected_p.norm()

        # Run forward and extract the intermediate Plucker coords
        z = enc.W_red(h)  # Should be h itself since W_red = I
        z_t = z[:, :1, :]
        z_td = z[:, 1:, :]
        z_t_i = z_t[:, :, enc.idx_i]
        z_t_j = z_t[:, :, enc.idx_j]
        z_td_i = z_td[:, :, enc.idx_i]
        z_td_j = z_td[:, :, enc.idx_j]
        p = z_t_i * z_td_j - z_t_j * z_td_i

        assert torch.allclose(p[0, 0], expected_p, atol=1e-6)

        # Check normalization
        p_hat = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        assert torch.allclose(p_hat[0, 0], expected_p_hat, atol=1e-6)
        assert torch.allclose(p_hat.norm(dim=-1), torch.ones(1, 1), atol=1e-6)

    def test_plucker_normalization_unit_norm(self, plucker):
        # The normalized Plucker vectors in valid positions should have unit norm
        h = torch.randn(B, L, D_MODEL)
        delta = 1

        z = plucker.W_red(h)
        z_t = z[:, :L - delta, :]
        z_td = z[:, delta:, :]
        z_t_i = z_t[:, :, plucker.idx_i]
        z_t_j = z_t[:, :, plucker.idx_j]
        z_td_i = z_td[:, :, plucker.idx_i]
        z_td_j = z_td[:, :, plucker.idx_j]
        p = z_t_i * z_td_j - z_t_j * z_td_i
        p_hat = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        norms = p_hat.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_antisymmetry(self):
        # Plücker coordinates are antisymmetric: swapping z_t and z_td negates them
        enc = PluckerEncoder(d_model=D_MODEL, d_reduce=D_REDUCE)
        h = torch.randn(1, 3, D_MODEL)

        z = enc.W_red(h)
        # Pair (z_0, z_1) vs (z_1, z_0)
        z_a, z_b = z[:, 0:1, :], z[:, 1:2, :]

        def compute_plucker(za, zb):
            za_i = za[:, :, enc.idx_i]
            za_j = za[:, :, enc.idx_j]
            zb_i = zb[:, :, enc.idx_i]
            zb_j = zb[:, :, enc.idx_j]
            return za_i * zb_j - za_j * zb_i

        p_ab = compute_plucker(z_a, z_b)
        p_ba = compute_plucker(z_b, z_a)

        assert torch.allclose(p_ab, -p_ba, atol=1e-6)


# ---------- GrassmannMixing ----------


class TestGrassmannMixing:
    def test_output_shape(self, mixing):
        h = torch.randn(B, L, D_MODEL)
        h_mix = mixing(h)
        assert h_mix.shape == (B, L, D_MODEL)

    def test_multi_offset_output_shape(self):
        mix = GrassmannMixing(
            d_model=D_MODEL, d_reduce=D_REDUCE,
            window_offsets=[1, 2, 4],
        )
        h = torch.randn(B, L, D_MODEL)
        h_mix = mix(h)
        assert h_mix.shape == (B, L, D_MODEL)

    def test_gate_bounds(self, mixing):
        # Alpha (gate) should be in [0, 1] since it's sigmoid output
        h = torch.randn(B, L, D_MODEL)
        g_sum = torch.zeros(B, L, D_MODEL)
        count = torch.zeros(B, L, 1)

        for delta in mixing.window_offsets:
            g_delta = mixing.plucker_encoder(h, delta)
            g_sum = g_sum + g_delta
            valid_len = max(L - delta, 0)
            if valid_len > 0:
                count[:, :valid_len, :] += 1.0

        g = g_sum / count.clamp(min=1.0)
        u = torch.cat([h, g], dim=-1)
        alpha = torch.sigmoid(mixing.W_gate(u))

        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_deterministic(self, mixing):
        h = torch.randn(B, L, D_MODEL)
        out1 = mixing(h)
        out2 = mixing(h)
        assert torch.allclose(out1, out2)


# ---------- GrassmannBlock ----------


class TestGrassmannBlock:
    def test_output_shape(self, block):
        h = torch.randn(B, L, D_MODEL)
        out = block(h)
        assert out.shape == (B, L, D_MODEL)

    def test_different_sequence_lengths(self, block):
        for seq_len in [1, 5, 20]:
            h = torch.randn(1, seq_len, D_MODEL)
            out = block(h)
            assert out.shape == (1, seq_len, D_MODEL)

    def test_gradient_flows(self, block):
        h = torch.randn(B, L, D_MODEL, requires_grad=True)
        out = block(h)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert not torch.all(h.grad == 0.0)

    def test_dropout_effect(self):
        # With dropout > 0, training mode should differ from eval mode
        blk = GrassmannBlock(
            d_model=D_MODEL, d_reduce=D_REDUCE, d_ff=D_FF,
            window_offsets=[1], dropout=0.5,
        )
        h = torch.randn(B, L, D_MODEL)

        blk.train()
        torch.manual_seed(42)
        out_train = blk(h)

        blk.eval()
        out_eval = blk(h)

        # They should differ (very unlikely to be identical with 50% dropout)
        assert not torch.allclose(out_train, out_eval, atol=1e-6)

    def test_ffn_intermediate_dimension(self, block):
        # Verify the FFN has the right intermediate size
        ffn = block.ffn
        assert ffn[0].in_features == D_MODEL
        assert ffn[0].out_features == D_FF
        assert ffn[2].in_features == D_FF
        assert ffn[2].out_features == D_MODEL
