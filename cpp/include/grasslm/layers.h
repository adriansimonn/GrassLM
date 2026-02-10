#pragma once

#include <grasslm/tensor.h>

#include <vector>

namespace grasslm {

/// Plücker coordinate encoder: projects token pairs onto Gr(2, r).
class PluckerEncoder {
public:
    PluckerEncoder() = default;

    /// Initialize from loaded weight tensors.
    void init(const Tensor& W_red_weight, const Tensor& W_red_bias,
              const Tensor& W_plu_weight, const Tensor& W_plu_bias,
              int d_model, int d_reduce);

    /// Forward pass: compute Plücker features for pairs at the given offset.
    /// h shape: (L, d_model), returns shape: (L, d_model)
    Tensor forward(const Tensor& h, int offset) const;

private:
    Tensor W_red_weight_;  // (d_reduce, d_model)
    Tensor W_red_bias_;    // (d_reduce,)
    Tensor W_plu_weight_;  // (d_model, plucker_dim)
    Tensor W_plu_bias_;    // (d_model,)
    int d_model_ = 0;
    int d_reduce_ = 0;
    int plucker_dim_ = 0;

    // Precomputed index pairs (i, j) with i < j
    std::vector<int> idx_i_;
    std::vector<int> idx_j_;
};

/// Grassmann mixing with gated fusion across multiple window offsets.
class GrassmannMixing {
public:
    GrassmannMixing() = default;

    void init(const PluckerEncoder& encoder,
              const Tensor& W_gate_weight, const Tensor& W_gate_bias,
              const std::vector<int>& window_offsets, int d_model);

    /// Forward pass. h shape: (L, d_model), returns shape: (L, d_model)
    Tensor forward(const Tensor& h) const;

private:
    PluckerEncoder plucker_encoder_;
    Tensor W_gate_weight_;  // (d_model, 2*d_model)
    Tensor W_gate_bias_;    // (d_model,)
    std::vector<int> window_offsets_;
    int d_model_ = 0;
};

/// Full Grassmann block: mixing → LN → FFN → residual → LN.
class GrassmannBlock {
public:
    GrassmannBlock() = default;

    void init(const GrassmannMixing& mixing,
              const Tensor& ln1_weight, const Tensor& ln1_bias,
              const Tensor& ffn_w1_weight, const Tensor& ffn_w1_bias,
              const Tensor& ffn_w2_weight, const Tensor& ffn_w2_bias,
              const Tensor& ln2_weight, const Tensor& ln2_bias,
              int d_model, int d_ff);

    /// Forward pass. h shape: (L, d_model), returns shape: (L, d_model)
    Tensor forward(const Tensor& h) const;

private:
    GrassmannMixing mixing_;
    Tensor ln1_weight_, ln1_bias_;
    Tensor ffn_w1_weight_, ffn_w1_bias_;  // (d_ff, d_model), (d_ff,)
    Tensor ffn_w2_weight_, ffn_w2_bias_;  // (d_model, d_ff), (d_model,)
    Tensor ln2_weight_, ln2_bias_;
    int d_model_ = 0;
    int d_ff_ = 0;
};

}  // namespace grasslm
