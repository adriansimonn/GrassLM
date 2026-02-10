#include <grasslm/generator.h>

namespace grasslm {

Generator::Generator(const GrassLMModel& model, const Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

std::string Generator::generate(const std::string& prompt,
                                const GenerationConfig& config) const {
    // TODO: Full implementation in step 2.6
    return "";
}

void Generator::generate_stream(const std::string& prompt,
                                const GenerationConfig& config,
                                TokenCallback callback) const {
    // TODO: Full implementation in step 2.6
}

int Generator::sample_token(const Tensor& logits, const GenerationConfig& config,
                            std::mt19937& rng) const {
    // TODO: Full implementation in step 2.6
    return 0;
}

}  // namespace grasslm
