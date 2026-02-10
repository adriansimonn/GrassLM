#pragma once

#include <grasslm/tensor.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace grasslm {

/// Model configuration parsed from the .grasslm binary header.
struct ModelConfig {
    uint32_t n_layers = 0;
    uint32_t d_model = 0;
    uint32_t d_reduce = 0;
    uint32_t d_ff = 0;
    uint32_t vocab_size = 0;
    uint32_t max_seq_len = 0;
    uint32_t dtype = 0;  // 0 = float32, 1 = float16
};

/// Loads model weights from the .grasslm binary format.
class WeightLoader {
public:
    /// Load weights from a .grasslm file. Returns true on success.
    bool load(const std::string& path);

    /// Access parsed model configuration.
    const ModelConfig& config() const { return config_; }

    /// Access loaded weights by name.
    const std::unordered_map<std::string, Tensor>& weights() const { return weights_; }

    /// Get a specific weight tensor by name. Throws if not found.
    const Tensor& get(const std::string& name) const;

private:
    ModelConfig config_;
    std::unordered_map<std::string, Tensor> weights_;
};

}  // namespace grasslm
