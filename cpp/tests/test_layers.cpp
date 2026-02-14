#include <grasslm/layers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace grasslm;

// --- Helper: create a PluckerEncoder with small dimensions ---
static PluckerEncoder make_test_encoder(int d_model, int d_reduce) {
    int plucker_dim = d_reduce * (d_reduce - 1) / 2;
    PluckerEncoder encoder;

    // Initialize with small random-ish weights (deterministic)
    Tensor W_red({d_reduce, d_model});
    Tensor W_red_b({d_reduce});
    Tensor W_plu({d_model, plucker_dim});
    Tensor W_plu_b({d_model});

    // Fill with a simple pattern to get non-trivial outputs
    for (int i = 0; i < W_red.numel(); ++i) W_red(i) = 0.1f * ((i % 7) - 3);
    for (int i = 0; i < W_red_b.numel(); ++i) W_red_b(i) = 0.01f * i;
    for (int i = 0; i < W_plu.numel(); ++i) W_plu(i) = 0.05f * ((i % 5) - 2);
    for (int i = 0; i < W_plu_b.numel(); ++i) W_plu_b(i) = 0.01f * i;

    encoder.init(W_red, W_red_b, W_plu, W_plu_b, d_model, d_reduce);
    return encoder;
}

// --- PluckerEncoder tests ---

TEST(PluckerEncoderTest, IndexPairGeneration) {
    PluckerEncoder encoder;
    // d_reduce=4: pairs are (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) = 6 pairs
    Tensor W_red({4, 8});
    Tensor W_red_b({4});
    Tensor W_plu({8, 6});
    Tensor W_plu_b({8});
    encoder.init(W_red, W_red_b, W_plu, W_plu_b, 8, 4);
    SUCCEED();
}

TEST(PluckerEncoderTest, ForwardOutputShape) {
    int d_model = 8, d_reduce = 4;
    PluckerEncoder encoder = make_test_encoder(d_model, d_reduce);

    Tensor h({4, d_model});
    for (int i = 0; i < h.numel(); ++i) h(i) = 0.1f * (i % 11);

    Tensor g = encoder.forward(h, 1);
    EXPECT_EQ(g.size(0), 4);
    EXPECT_EQ(g.size(1), d_model);
}

TEST(PluckerEncoderTest, ForwardZeroPadding) {
    // With offset=1, the first position should be zero-padded (no backward neighbor)
    int d_model = 8, d_reduce = 4;
    PluckerEncoder encoder = make_test_encoder(d_model, d_reduce);

    Tensor h({4, d_model});
    for (int i = 0; i < h.numel(); ++i) h(i) = 0.1f * (i % 11);

    Tensor g = encoder.forward(h, 1);

    // First row (position 0) should be all zeros (no valid pair at t-1 when t=0)
    for (int j = 0; j < d_model; ++j) {
        EXPECT_FLOAT_EQ(g(0, j), 0.0f);
    }
}

TEST(PluckerEncoderTest, ForwardLargeOffsetReturnsZeros) {
    int d_model = 8, d_reduce = 4;
    PluckerEncoder encoder = make_test_encoder(d_model, d_reduce);

    Tensor h({4, d_model});
    h.fill(1.0f);

    // Offset >= L: all output should be zeros
    Tensor g = encoder.forward(h, 4);
    for (int i = 0; i < g.numel(); ++i) {
        EXPECT_FLOAT_EQ(g(i), 0.0f);
    }
}

TEST(PluckerEncoderTest, PluckerAntisymmetry) {
    // If z_t == z_td (offset=0 pairing token with itself), Plucker coords
    // should be zero because p[i,j] = z[i]*z[j] - z[j]*z[i] = 0
    int d_model = 8, d_reduce = 4;

    // Use identity-like W_red to pass input through
    Tensor W_red({d_reduce, d_model});
    W_red.zeros();
    for (int i = 0; i < d_reduce; ++i) W_red(i, i) = 1.0f;
    Tensor W_red_b({d_reduce});
    W_red_b.zeros();

    int plucker_dim = d_reduce * (d_reduce - 1) / 2;
    Tensor W_plu({d_model, plucker_dim});
    for (int i = 0; i < W_plu.numel(); ++i) W_plu(i) = 1.0f;
    Tensor W_plu_b({d_model});
    W_plu_b.zeros();

    PluckerEncoder encoder;
    encoder.init(W_red, W_red_b, W_plu, W_plu_b, d_model, d_reduce);

    // Create h where all rows are identical
    Tensor h({3, d_model});
    for (int t = 0; t < 3; ++t)
        for (int j = 0; j < d_model; ++j)
            h(t, j) = 0.5f * (j + 1);

    // With offset=1, consecutive tokens are identical → Plucker coords = 0
    // So g should be W_plu @ 0 + bias = 0 (bias is zero)
    Tensor g = encoder.forward(h, 1);
    for (int t = 1; t < 3; ++t) {  // valid positions (1..2, position 0 is zero-padded)
        for (int j = 0; j < d_model; ++j) {
            EXPECT_NEAR(g(t, j), 0.0f, 1e-5f);
        }
    }
}

// --- GrassmannMixing tests ---

TEST(GrassmannMixingTest, ForwardOutputShape) {
    int d_model = 8, d_reduce = 4;
    PluckerEncoder encoder = make_test_encoder(d_model, d_reduce);

    Tensor W_gate({d_model, 2 * d_model});
    Tensor W_gate_b({d_model});
    for (int i = 0; i < W_gate.numel(); ++i) W_gate(i) = 0.01f * ((i % 9) - 4);
    W_gate_b.zeros();

    GrassmannMixing mixing;
    mixing.init(encoder, W_gate, W_gate_b, {1}, d_model);

    Tensor h({4, d_model});
    for (int i = 0; i < h.numel(); ++i) h(i) = 0.1f * (i % 11);

    Tensor h_mix = mixing.forward(h);
    EXPECT_EQ(h_mix.size(0), 4);
    EXPECT_EQ(h_mix.size(1), d_model);
}

// --- GrassmannBlock tests ---

TEST(GrassmannBlockTest, ForwardOutputShape) {
    int d_model = 8, d_reduce = 4, d_ff = 16;
    PluckerEncoder encoder = make_test_encoder(d_model, d_reduce);

    Tensor W_gate({d_model, 2 * d_model});
    Tensor W_gate_b({d_model});
    for (int i = 0; i < W_gate.numel(); ++i) W_gate(i) = 0.01f * ((i % 9) - 4);
    W_gate_b.zeros();

    GrassmannMixing mixing;
    mixing.init(encoder, W_gate, W_gate_b, {1}, d_model);

    // Layer norm weights (gamma=1, beta=0)
    Tensor ln1_w({d_model}), ln1_b({d_model});
    Tensor ln2_w({d_model}), ln2_b({d_model});
    ln1_w.fill(1.0f); ln1_b.zeros();
    ln2_w.fill(1.0f); ln2_b.zeros();

    // FFN weights
    Tensor ffn_w1({d_ff, d_model}), ffn_b1({d_ff});
    Tensor ffn_w2({d_model, d_ff}), ffn_b2({d_model});
    for (int i = 0; i < ffn_w1.numel(); ++i) ffn_w1(i) = 0.02f * ((i % 7) - 3);
    ffn_b1.zeros();
    for (int i = 0; i < ffn_w2.numel(); ++i) ffn_w2(i) = 0.02f * ((i % 5) - 2);
    ffn_b2.zeros();

    GrassmannBlock block;
    block.init(mixing, ln1_w, ln1_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2,
               ln2_w, ln2_b, d_model, d_ff);

    Tensor h({4, d_model});
    for (int i = 0; i < h.numel(); ++i) h(i) = 0.1f * (i % 11);

    Tensor out = block.forward(h);
    EXPECT_EQ(out.size(0), 4);
    EXPECT_EQ(out.size(1), d_model);

    // Output should be finite and non-trivial
    bool all_zero = true;
    bool any_nan = false;
    for (int i = 0; i < out.numel(); ++i) {
        if (out(i) != 0.0f) all_zero = false;
        if (std::isnan(out(i))) any_nan = true;
    }
    EXPECT_FALSE(all_zero);
    EXPECT_FALSE(any_nan);
}

TEST(GrassmannBlockTest, LayerNormOutput) {
    // After a full block pass, each row of the output should have
    // approximately zero mean (due to final layer norm with gamma=1, beta=0)
    int d_model = 8, d_reduce = 4, d_ff = 16;
    PluckerEncoder encoder = make_test_encoder(d_model, d_reduce);

    Tensor W_gate({d_model, 2 * d_model});
    Tensor W_gate_b({d_model});
    for (int i = 0; i < W_gate.numel(); ++i) W_gate(i) = 0.01f * ((i % 9) - 4);
    W_gate_b.zeros();

    GrassmannMixing mixing;
    mixing.init(encoder, W_gate, W_gate_b, {1}, d_model);

    Tensor ln1_w({d_model}), ln1_b({d_model});
    Tensor ln2_w({d_model}), ln2_b({d_model});
    ln1_w.fill(1.0f); ln1_b.zeros();
    ln2_w.fill(1.0f); ln2_b.zeros();

    Tensor ffn_w1({d_ff, d_model}), ffn_b1({d_ff});
    Tensor ffn_w2({d_model, d_ff}), ffn_b2({d_model});
    for (int i = 0; i < ffn_w1.numel(); ++i) ffn_w1(i) = 0.02f * ((i % 7) - 3);
    ffn_b1.zeros();
    for (int i = 0; i < ffn_w2.numel(); ++i) ffn_w2(i) = 0.02f * ((i % 5) - 2);
    ffn_b2.zeros();

    GrassmannBlock block;
    block.init(mixing, ln1_w, ln1_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2,
               ln2_w, ln2_b, d_model, d_ff);

    Tensor h({4, d_model});
    for (int i = 0; i < h.numel(); ++i) h(i) = 0.1f * (i % 11);

    Tensor out = block.forward(h);

    // Each row should have approximately zero mean due to layer norm
    for (int t = 0; t < 4; ++t) {
        float row_mean = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            row_mean += out(t, j);
        }
        row_mean /= d_model;
        EXPECT_NEAR(row_mean, 0.0f, 1e-4f);
    }
}
