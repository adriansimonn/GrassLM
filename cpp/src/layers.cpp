#include <grasslm/layers.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace grasslm {

// --- PluckerEncoder ---

void PluckerEncoder::init(const Tensor& W_red_weight, const Tensor& W_red_bias,
                          const Tensor& W_plu_weight, const Tensor& W_plu_bias,
                          int d_model, int d_reduce) {
    W_red_weight_ = W_red_weight;
    W_red_bias_ = W_red_bias;
    W_plu_weight_ = W_plu_weight;
    W_plu_bias_ = W_plu_bias;
    d_model_ = d_model;
    d_reduce_ = d_reduce;
    plucker_dim_ = d_reduce * (d_reduce - 1) / 2;

    // Precompute index pairs (i, j) with i < j
    idx_i_.clear();
    idx_j_.clear();
    for (int i = 0; i < d_reduce; ++i) {
        for (int j = i + 1; j < d_reduce; ++j) {
            idx_i_.push_back(i);
            idx_j_.push_back(j);
        }
    }
}

Tensor PluckerEncoder::forward(const Tensor& h, int offset) const {
    int L = h.size(0);

    // 1. Reduce to low-rank space: z = h @ W_red^T + bias
    Tensor z = h.linear(W_red_weight_, W_red_bias_);  // (L, d_reduce)

    // 2. No valid pairs if offset >= L
    if (offset >= L) {
        Tensor result({L, d_model_});
        result.zeros();
        return result;
    }

    int valid_len = L - offset;

    // 3. Form causal pairs: z_t paired with z_{t-offset} (look backward)
    Tensor z_t = z.slice(0, offset, L);           // (valid_len, d_reduce)  — positions offset..L-1
    Tensor z_td = z.slice(0, 0, valid_len);       // (valid_len, d_reduce)  — positions 0..L-offset-1

    // 4. Compute Plucker coordinates:
    //    p[l][k] = z_t[l][idx_i[k]] * z_td[l][idx_j[k]]
    //            - z_t[l][idx_j[k]] * z_td[l][idx_i[k]]
    Tensor p({valid_len, plucker_dim_});
    for (int l = 0; l < valid_len; ++l) {
        for (int k = 0; k < plucker_dim_; ++k) {
            int ii = idx_i_[k];
            int jj = idx_j_[k];
            p(l, k) = z_t(l, ii) * z_td(l, jj) - z_t(l, jj) * z_td(l, ii);
        }
    }

    // 5. Normalize: p_hat = p / ||p||_2 (per row, clamped)
    Tensor p_norm = p.norm(-1);  // (valid_len, 1)
    Tensor p_norm_clamped = p_norm.clamp(1e-8f, 1e30f);

    Tensor p_hat({valid_len, plucker_dim_});
    for (int l = 0; l < valid_len; ++l) {
        float inv_norm = 1.0f / p_norm_clamped(l, 0);
        for (int k = 0; k < plucker_dim_; ++k) {
            p_hat(l, k) = p(l, k) * inv_norm;
        }
    }

    // 6. Project back to model dimension: g_valid = p_hat @ W_plu^T + bias
    Tensor g_valid = p_hat.linear(W_plu_weight_, W_plu_bias_);  // (valid_len, d_model)

    // 7. Pad to full sequence length (first offset positions have no backward neighbor)
    Tensor g({L, d_model_});
    g.zeros();
    float* g_data = g.data();
    const float* gv_data = g_valid.data();
    for (int l = 0; l < valid_len; ++l) {
        std::copy(gv_data + l * d_model_, gv_data + (l + 1) * d_model_,
                  g_data + (l + offset) * d_model_);
    }

    return g;
}

// --- GrassmannMixing ---

void GrassmannMixing::init(const PluckerEncoder& encoder,
                           const Tensor& W_gate_weight, const Tensor& W_gate_bias,
                           const std::vector<int>& window_offsets, int d_model) {
    plucker_encoder_ = encoder;
    W_gate_weight_ = W_gate_weight;
    W_gate_bias_ = W_gate_bias;
    window_offsets_ = window_offsets;
    d_model_ = d_model;
}

Tensor GrassmannMixing::forward(const Tensor& h) const {
    int L = h.size(0);
    int d = d_model_;

    // 1. Compute Plucker features for each offset and accumulate
    Tensor g_sum({L, d});
    g_sum.zeros();
    std::vector<float> count(L, 0.0f);

    for (int delta : window_offsets_) {
        Tensor g_delta = plucker_encoder_.forward(h, delta);  // (L, d)
        g_sum.axpy(1.0f, g_delta);

        // Positions delta..L-1 are valid (they have a backward neighbor)
        for (int t = delta; t < L; ++t) {
            count[t] += 1.0f;
        }
    }

    // 2. Average over valid offsets
    Tensor g({L, d});
    float* g_data = g.data();
    const float* gs_data = g_sum.data();
    for (int t = 0; t < L; ++t) {
        float c = std::max(count[t], 1.0f);
        for (int j = 0; j < d; ++j) {
            g_data[t * d + j] = gs_data[t * d + j] / c;
        }
    }

    // 3. Gated fusion: alpha = sigmoid(W_gate([h; g]))
    Tensor u = Tensor::cat({h, g}, 1);                     // (L, 2d)
    Tensor alpha = u.linear(W_gate_weight_, W_gate_bias_);  // (L, d)
    alpha = alpha.sigmoid();

    // 4. h_mix = alpha * h + (1 - alpha) * g
    Tensor h_mix({L, d});
    float* hm_data = h_mix.data();
    const float* h_data = h.data();
    const float* a_data = alpha.data();
    g_data = g.data();
    for (int i = 0; i < L * d; ++i) {
        hm_data[i] = a_data[i] * h_data[i] + (1.0f - a_data[i]) * g_data[i];
    }

    return h_mix;
}

// --- GrassmannBlock ---

void GrassmannBlock::init(const GrassmannMixing& mixing,
                          const Tensor& ln1_weight, const Tensor& ln1_bias,
                          const Tensor& ffn_w1_weight, const Tensor& ffn_w1_bias,
                          const Tensor& ffn_w2_weight, const Tensor& ffn_w2_bias,
                          const Tensor& ln2_weight, const Tensor& ln2_bias,
                          int d_model, int d_ff) {
    mixing_ = mixing;
    ln1_weight_ = ln1_weight;
    ln1_bias_ = ln1_bias;
    ffn_w1_weight_ = ffn_w1_weight;
    ffn_w1_bias_ = ffn_w1_bias;
    ffn_w2_weight_ = ffn_w2_weight;
    ffn_w2_bias_ = ffn_w2_bias;
    ln2_weight_ = ln2_weight;
    ln2_bias_ = ln2_bias;
    d_model_ = d_model;
    d_ff_ = d_ff;
}

Tensor GrassmannBlock::forward(const Tensor& h) const {
    // Grassmann mixing (no dropout at inference)
    Tensor h_mix = mixing_.forward(h);

    // Layer norm 1
    h_mix = h_mix.layernorm(ln1_weight_, ln1_bias_);

    // FFN: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
    Tensor ffn_hidden = h_mix.linear(ffn_w1_weight_, ffn_w1_bias_);  // (L, d_ff)
    ffn_hidden = ffn_hidden.gelu();
    Tensor ffn_out = ffn_hidden.linear(ffn_w2_weight_, ffn_w2_bias_);  // (L, d_model)

    // Residual connection around FFN + Layer norm 2
    Tensor h_out = (h_mix + ffn_out).layernorm(ln2_weight_, ln2_bias_);

    return h_out;
}

}  // namespace grasslm
