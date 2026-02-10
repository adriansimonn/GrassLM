#include <grasslm/model.h>

#include <stdexcept>
#include <string>

namespace grasslm {

// Window schedule matching the Python model (model.py WINDOW_SCHEDULES)
static std::vector<int> get_window_schedule(int n_layers) {
    if (n_layers == 6) {
        return {1, 2, 4, 8, 12, 16};
    }
    if (n_layers == 12) {
        return {1, 1, 2, 2, 4, 4, 8, 8, 12, 12, 16, 16};
    }
    return std::vector<int>(n_layers, 1);
}

bool GrassLMModel::load(const WeightLoader& loader) {
    config_ = loader.config();

    try {
        // Load embeddings
        tok_embed_ = loader.get("tok_embed.weight");  // (vocab_size, d_model)
        pos_embed_ = loader.get("pos_embed.weight");  // (max_seq_len, d_model)

        // Window schedule
        int n_layers = static_cast<int>(config_.n_layers);
        std::vector<int> schedule = get_window_schedule(n_layers);

        // Load blocks
        blocks_.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            std::string prefix = "blocks." + std::to_string(i) + ".";

            // Plucker encoder
            PluckerEncoder encoder;
            encoder.init(
                loader.get(prefix + "mixing.plucker_encoder.W_red.weight"),
                loader.get(prefix + "mixing.plucker_encoder.W_red.bias"),
                loader.get(prefix + "mixing.plucker_encoder.W_plu.weight"),
                loader.get(prefix + "mixing.plucker_encoder.W_plu.bias"),
                static_cast<int>(config_.d_model),
                static_cast<int>(config_.d_reduce));

            // Grassmann mixing
            GrassmannMixing mixing;
            mixing.init(
                encoder,
                loader.get(prefix + "mixing.W_gate.weight"),
                loader.get(prefix + "mixing.W_gate.bias"),
                {schedule[i]},
                static_cast<int>(config_.d_model));

            // Full block
            blocks_[i].init(
                mixing,
                loader.get(prefix + "ln1.weight"),
                loader.get(prefix + "ln1.bias"),
                loader.get(prefix + "ffn.0.weight"),
                loader.get(prefix + "ffn.0.bias"),
                loader.get(prefix + "ffn.2.weight"),
                loader.get(prefix + "ffn.2.bias"),
                loader.get(prefix + "ln2.weight"),
                loader.get(prefix + "ln2.bias"),
                static_cast<int>(config_.d_model),
                static_cast<int>(config_.d_ff));
        }

        // Final layer norm
        ln_final_weight_ = loader.get("ln_final.weight");
        ln_final_bias_ = loader.get("ln_final.bias");

    } catch (const std::runtime_error&) {
        return false;
    }

    return true;
}

Tensor GrassLMModel::forward(const std::vector<int>& token_ids) const {
    int L = static_cast<int>(token_ids.size());
    int d = static_cast<int>(config_.d_model);

    // Token + positional embeddings
    Tensor h({L, d});
    for (int t = 0; t < L; ++t) {
        int id = token_ids[t];
        for (int j = 0; j < d; ++j) {
            h(t, j) = tok_embed_(id, j) + pos_embed_(t, j);
        }
    }

    // Pass through Grassmann blocks
    for (const auto& block : blocks_) {
        h = block.forward(h);
    }

    // Final layer norm
    h = h.layernorm(ln_final_weight_, ln_final_bias_);

    // LM head (weight-tied with tok_embed): logits = h @ tok_embed^T
    Tensor logits = h.linear(tok_embed_);  // (L, vocab_size)

    return logits;
}

}  // namespace grasslm
