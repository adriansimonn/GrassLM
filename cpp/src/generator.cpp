#include <grasslm/generator.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace grasslm {

Generator::Generator(const GrassLMModel& model, const Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

int Generator::sample_token(const Tensor& logits, const GenerationConfig& config,
                            std::mt19937& rng) const {
    int vocab_size = logits.numel();
    const float* data = logits.data();

    // Greedy: temperature <= 0 or very small
    if (config.temperature <= 0.0f) {
        return static_cast<int>(
            std::max_element(data, data + vocab_size) - data);
    }

    // Copy logits and apply temperature scaling
    std::vector<float> scaled(data, data + vocab_size);
    if (config.temperature != 1.0f) {
        float inv_temp = 1.0f / config.temperature;
        for (float& v : scaled) {
            v *= inv_temp;
        }
    }

    // Top-k filtering: keep only the top_k highest logits
    if (config.top_k > 0 && config.top_k < vocab_size) {
        // Find the k-th largest value as threshold
        std::vector<float> sorted_vals(scaled);
        std::nth_element(sorted_vals.begin(),
                         sorted_vals.begin() + config.top_k,
                         sorted_vals.end(),
                         std::greater<float>());
        float threshold = sorted_vals[config.top_k];

        for (float& v : scaled) {
            if (v < threshold) {
                v = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // Softmax
    float max_val = *std::max_element(scaled.begin(), scaled.end());
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(scaled[i] - max_val);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (float& p : probs) {
        p *= inv_sum;
    }

    // Top-p (nucleus) filtering
    if (config.top_p < 1.0f) {
        // Sort indices by probability descending
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&probs](int a, int b) { return probs[a] > probs[b]; });

        // Find cutoff where cumulative probability exceeds top_p
        float cumulative = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < indices.size(); i++) {
            cumulative += probs[indices[i]];
            if (cumulative > config.top_p) {
                cutoff = i + 1;  // keep this token (first to exceed)
                break;
            }
        }

        // Zero out tokens below the cutoff
        for (size_t i = cutoff; i < indices.size(); i++) {
            probs[indices[i]] = 0.0f;
        }

        // Re-normalize
        sum = 0.0f;
        for (float p : probs) sum += p;
        if (sum > 0.0f) {
            inv_sum = 1.0f / sum;
            for (float& p : probs) p *= inv_sum;
        }
    }

    // Sample from the probability distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

std::string Generator::generate(const std::string& prompt,
                                const GenerationConfig& config) const {
    std::string result;
    generate_stream(prompt, config, [&result](const std::string& token) {
        result += token;
    });
    return result;
}

void Generator::generate_stream(const std::string& prompt,
                                const GenerationConfig& config,
                                TokenCallback callback) const {
    // Encode prompt
    std::vector<int> token_ids = tokenizer_.encode(prompt);
    if (token_ids.empty()) return;

    int max_seq_len = static_cast<int>(model_.config().max_seq_len);

    // Seed RNG
    std::random_device rd;
    std::mt19937 rng(rd());

    bool first_generated = true;

    for (int step = 0; step < config.max_tokens; step++) {
        // Truncate to max_seq_len if needed (keep most recent tokens)
        std::vector<int> context;
        if (static_cast<int>(token_ids.size()) > max_seq_len) {
            context.assign(token_ids.end() - max_seq_len, token_ids.end());
        } else {
            context = token_ids;
        }

        // Forward pass: get logits for all positions
        Tensor logits = model_.forward(context);  // (L, vocab_size)

        // Extract logits for the last position
        int L = logits.size(0);
        int V = logits.size(1);
        Tensor last_logits({V});
        const float* row = logits.data() + (L - 1) * V;
        std::copy(row, row + V, last_logits.data());

        // Sample next token
        int next_token = sample_token(last_logits, config, rng);

        // Check for EOS
        if (next_token == config.eos_token_id) break;

        // Decode token with proper spacing
        const std::string& raw_token = tokenizer_.id_to_token(next_token);

        // Skip special tokens
        if (raw_token == "[CLS]" || raw_token == "[SEP]" || raw_token == "[PAD]" ||
            raw_token == "[UNK]" || raw_token == "[MASK]") {
            token_ids.push_back(next_token);
            continue;
        }

        std::string text;
        if (raw_token.size() >= 2 && raw_token[0] == '#' && raw_token[1] == '#') {
            // Continuation subword: no space, strip ## prefix
            text = raw_token.substr(2);
        } else {
            // Regular token: prepend space unless it's the first generated token
            if (!first_generated) {
                text = " " + raw_token;
            } else {
                text = raw_token;
            }
        }

        first_generated = false;

        if (!text.empty()) {
            callback(text);
        }

        // Append to sequence
        token_ids.push_back(next_token);
    }
}

}  // namespace grasslm
