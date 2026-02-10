#pragma once

#include <grasslm/layers.h>
#include <grasslm/tensor.h>
#include <grasslm/weight_loader.h>

#include <string>
#include <vector>

namespace grasslm {

/// Full GrassLM causal language model: embeddings + N blocks + LM head.
class GrassLMModel {
public:
    GrassLMModel() = default;

    /// Load model from a WeightLoader (must already have loaded weights).
    bool load(const WeightLoader& loader);

    /// Forward pass: token IDs → logits.
    /// token_ids: vector of token indices, length L.
    /// Returns logits tensor of shape (L, vocab_size).
    Tensor forward(const std::vector<int>& token_ids) const;

    const ModelConfig& config() const { return config_; }

private:
    ModelConfig config_;

    // Embeddings
    Tensor tok_embed_;  // (vocab_size, d_model)
    Tensor pos_embed_;  // (max_seq_len, d_model)

    // Blocks
    std::vector<GrassmannBlock> blocks_;

    // Final layer norm
    Tensor ln_final_weight_;
    Tensor ln_final_bias_;

    // LM head is weight-tied with tok_embed_ (transpose for logits)
};

}  // namespace grasslm
