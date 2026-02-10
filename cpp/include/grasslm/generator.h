#pragma once

#include <grasslm/model.h>
#include <grasslm/tokenizer.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

namespace grasslm {

/// Configuration for text generation.
struct GenerationConfig {
    int max_tokens = 128;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    int eos_token_id = 102;  // [SEP] for BERT vocab
};

/// Callback invoked for each generated token (for streaming output).
using TokenCallback = std::function<void(const std::string& token)>;

/// Token-by-token text generator using GrassLMModel.
class Generator {
public:
    Generator(const GrassLMModel& model, const Tokenizer& tokenizer);

    /// Generate text from a prompt string.
    std::string generate(const std::string& prompt, const GenerationConfig& config) const;

    /// Streaming generation: calls callback for each new token.
    void generate_stream(const std::string& prompt,
                         const GenerationConfig& config,
                         TokenCallback callback) const;

private:
    const GrassLMModel& model_;
    const Tokenizer& tokenizer_;

    /// Sample a token from logits given the generation config.
    int sample_token(const Tensor& logits, const GenerationConfig& config,
                     std::mt19937& rng) const;
};

}  // namespace grasslm
