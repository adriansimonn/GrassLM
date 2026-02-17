#include "GrassLMBridge.h"

#include <grasslm/generator.h>
#include <grasslm/model.h>
#include <grasslm/tokenizer.h>
#include <grasslm/weight_loader.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/// Internal context holding all engine state.
struct GrassLMContextImpl {
    grasslm::WeightLoader loader;
    grasslm::GrassLMModel model;
    grasslm::Tokenizer tokenizer;
    grasslm::Generator* generator;

    GrassLMContextImpl() : generator(nullptr) {}
    ~GrassLMContextImpl() { delete generator; }
};

extern "C" {

GrassLMContext grasslm_create(const char* model_path, const char* vocab_path) {
    if (!model_path || !vocab_path) return nullptr;

    auto* ctx = new (std::nothrow) GrassLMContextImpl();
    if (!ctx) return nullptr;

    // Load model weights
    if (!ctx->loader.load(model_path)) {
        delete ctx;
        return nullptr;
    }

    // Initialize model from weights
    if (!ctx->model.load(ctx->loader)) {
        delete ctx;
        return nullptr;
    }

    // Load tokenizer vocabulary
    if (!ctx->tokenizer.load(vocab_path)) {
        delete ctx;
        return nullptr;
    }

    // Create generator
    ctx->generator = new (std::nothrow) grasslm::Generator(ctx->model, ctx->tokenizer);
    if (!ctx->generator) {
        delete ctx;
        return nullptr;
    }

    return static_cast<GrassLMContext>(ctx);
}

void grasslm_destroy(GrassLMContext handle) {
    if (!handle) return;
    auto* ctx = static_cast<GrassLMContextImpl*>(handle);
    delete ctx;
}

char* grasslm_generate(GrassLMContext handle, const char* prompt, int max_tokens,
                        float temperature, float top_p) {
    if (!handle || !prompt) return nullptr;

    auto* ctx = static_cast<GrassLMContextImpl*>(handle);

    grasslm::GenerationConfig config;
    config.max_tokens = max_tokens;
    config.temperature = temperature;
    config.top_p = top_p;

    std::string result = ctx->generator->generate(prompt, config);

    // Allocate a C string copy for the caller
    char* output = static_cast<char*>(std::malloc(result.size() + 1));
    if (!output) return nullptr;
    std::memcpy(output, result.c_str(), result.size() + 1);
    return output;
}

void grasslm_generate_stream(GrassLMContext handle, const char* prompt,
                              int max_tokens, float temperature, float top_p,
                              TokenCallback callback, void* user_data) {
    if (!handle || !prompt || !callback) return;

    auto* ctx = static_cast<GrassLMContextImpl*>(handle);

    grasslm::GenerationConfig config;
    config.max_tokens = max_tokens;
    config.temperature = temperature;
    config.top_p = top_p;

    ctx->generator->generate_stream(prompt, config,
        [callback, user_data](const std::string& token) {
            callback(token.c_str(), user_data);
        });
}

void grasslm_free_string(char* str) {
    std::free(str);
}

// ---------------------------------------------------------------------------
// JSON helpers (minimal, no external dependency)
// ---------------------------------------------------------------------------

/// Escape a string for JSON (handles quotes and backslashes).
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

/// Append a JSON array of floats to the stream.
static void json_float_array(std::ostringstream& ss, const float* data, int count) {
    ss << '[';
    for (int i = 0; i < count; ++i) {
        if (i > 0) ss << ',';
        ss << data[i];
    }
    ss << ']';
}

/// Allocate a malloc'd C string copy of a std::string (caller frees).
static char* to_c_string(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

/// Parse a JSON array of ints, e.g. "[1,2,3]", into a vector.
static std::vector<int> parse_int_array(const char* json) {
    std::vector<int> result;
    if (!json) return result;

    const char* p = json;
    // Skip to first '['
    while (*p && *p != '[') ++p;
    if (*p == '[') ++p;

    while (*p) {
        // Skip whitespace and commas
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') ++p;
        if (*p == ']' || *p == '\0') break;

        char* end = nullptr;
        long val = std::strtol(p, &end, 10);
        if (end == p) break;  // no valid int found
        result.push_back(static_cast<int>(val));
        p = end;
    }
    return result;
}

/// Compute L2 norm of a row in a (rows, cols) row-major tensor.
static float row_l2_norm(const float* data, int row, int cols) {
    const float* row_ptr = data + row * cols;
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        sum += row_ptr[j] * row_ptr[j];
    }
    return std::sqrt(sum);
}

/// Downsample a row from `cols` to `target_cols` by averaging adjacent values.
/// Writes `target_cols` floats into `out`.
static void downsample_row(const float* row, int cols, float* out, int target_cols) {
    float bin_size = static_cast<float>(cols) / static_cast<float>(target_cols);
    for (int i = 0; i < target_cols; ++i) {
        int start = static_cast<int>(i * bin_size);
        int end = static_cast<int>((i + 1) * bin_size);
        if (end > cols) end = cols;
        if (end <= start) end = start + 1;
        float sum = 0.0f;
        for (int j = start; j < end; ++j) {
            sum += row[j];
        }
        out[i] = sum / static_cast<float>(end - start);
    }
}

/// Write a heatmap tensor (L x d_model) to JSON, downsampling columns if needed.
/// Appends to the ostringstream as a flat JSON array of floats.
/// Returns the number of columns actually written (may be < d_model).
static int json_heatmap(std::ostringstream& ss, const float* data, int L, int d_model, int max_cols) {
    int out_cols = std::min(d_model, max_cols);
    bool need_downsample = (d_model > max_cols);

    ss << '[';
    std::vector<float> ds_buf;
    if (need_downsample) ds_buf.resize(out_cols);

    for (int t = 0; t < L; ++t) {
        if (t > 0) ss << ',';
        if (need_downsample) {
            downsample_row(data + t * d_model, d_model, ds_buf.data(), out_cols);
            json_float_array(ss, ds_buf.data(), out_cols);
        } else {
            json_float_array(ss, data + t * d_model, out_cols);
        }
    }
    ss << ']';
    return out_cols;
}

// ---------------------------------------------------------------------------
// Interpretability API implementations
// ---------------------------------------------------------------------------

char* grasslm_tokenize(GrassLMContext handle, const char* text) {
    if (!handle || !text) return nullptr;

    auto* ctx = static_cast<GrassLMContextImpl*>(handle);

    std::vector<int> ids = ctx->tokenizer.encode(text);

    std::ostringstream ss;
    ss << "{\"tokens\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) ss << ',';
        ss << '"' << json_escape(ctx->tokenizer.id_to_token(ids[i])) << '"';
    }
    ss << "],\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) ss << ',';
        ss << ids[i];
    }
    ss << "]}";

    return to_c_string(ss.str());
}

char* grasslm_forward_debug(GrassLMContext handle, const char* token_ids_json) {
    if (!handle || !token_ids_json) return nullptr;

    auto* ctx = static_cast<GrassLMContextImpl*>(handle);
    std::vector<int> token_ids = parse_int_array(token_ids_json);
    if (token_ids.empty()) return nullptr;

    // Clamp to max_seq_len
    int max_len = static_cast<int>(ctx->model.config().max_seq_len);
    if (static_cast<int>(token_ids.size()) > max_len) {
        token_ids.resize(max_len);
    }

    grasslm::ForwardDebugResult dbg = ctx->model.forward_debug(token_ids);

    int L = static_cast<int>(token_ids.size());
    int d = static_cast<int>(ctx->model.config().d_model);
    int n_layers = static_cast<int>(ctx->model.config().n_layers);
    int vocab_size = static_cast<int>(ctx->model.config().vocab_size);
    const int MAX_HEATMAP_COLS = 128;

    std::ostringstream ss;
    ss << std::setprecision(6);
    ss << "{";

    // Metadata
    ss << "\"n_layers\":" << n_layers
       << ",\"d_model\":" << d
       << ",\"seq_len\":" << L;

    // embed_norms: L2 norm per position from embed_output
    ss << ",\"embed_norms\":[";
    for (int t = 0; t < L; ++t) {
        if (t > 0) ss << ',';
        ss << row_l2_norm(dbg.embed_output.data(), t, d);
    }
    ss << ']';

    // block_norms: N_layers arrays, each with L floats
    ss << ",\"block_norms\":[";
    for (int layer = 0; layer < n_layers; ++layer) {
        if (layer > 0) ss << ',';
        ss << '[';
        for (int t = 0; t < L; ++t) {
            if (t > 0) ss << ',';
            ss << row_l2_norm(dbg.block_outputs[layer].data(), t, d);
        }
        ss << ']';
    }
    ss << ']';

    // embed_heatmap: L x out_cols (downsampled)
    ss << ",\"embed_heatmap\":";
    int out_cols = json_heatmap(ss, dbg.embed_output.data(), L, d, MAX_HEATMAP_COLS);

    // heatmap_cols: actual number of columns in heatmap arrays
    ss << ",\"heatmap_cols\":" << out_cols;

    // block_heatmaps: N arrays, each L x out_cols
    ss << ",\"block_heatmaps\":[";
    for (int layer = 0; layer < n_layers; ++layer) {
        if (layer > 0) ss << ',';
        json_heatmap(ss, dbg.block_outputs[layer].data(), L, d, MAX_HEATMAP_COLS);
    }
    ss << ']';

    // final_norm_heatmap: L x out_cols
    ss << ",\"final_norm_heatmap\":";
    json_heatmap(ss, dbg.final_norm_output.data(), L, d, MAX_HEATMAP_COLS);

    // top_logits: softmax on last position, top 5 predictions
    // top_logits_per_position: per-position softmax top 5 predictions
    {
        std::vector<float> probs(vocab_size);
        std::vector<int> top_indices(vocab_size);
        const int top_k = 5;

        // Last-position predictions (backward-compatible top_logits)
        const float* last_logits = dbg.logits.data() + (L - 1) * vocab_size;
        float max_val = *std::max_element(last_logits, last_logits + vocab_size);
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(last_logits[i] - max_val);
            sum_exp += probs[i];
        }
        for (int i = 0; i < vocab_size; ++i) probs[i] /= sum_exp;

        for (int i = 0; i < vocab_size; ++i) top_indices[i] = i;
        std::partial_sort(top_indices.begin(), top_indices.begin() + top_k, top_indices.end(),
            [&probs](int a, int b) { return probs[a] > probs[b]; });

        ss << ",\"top_logits\":[";
        for (int i = 0; i < top_k && i < vocab_size; ++i) {
            if (i > 0) ss << ',';
            int idx = top_indices[i];
            ss << "{\"token\":\"" << json_escape(ctx->tokenizer.id_to_token(idx))
               << "\",\"prob\":" << probs[idx] << '}';
        }
        ss << ']';

        // Per-position predictions
        ss << ",\"top_logits_per_position\":[";
        for (int pos = 0; pos < L; ++pos) {
            if (pos > 0) ss << ',';
            const float* pos_logits = dbg.logits.data() + pos * vocab_size;

            max_val = *std::max_element(pos_logits, pos_logits + vocab_size);
            sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; ++i) {
                probs[i] = std::exp(pos_logits[i] - max_val);
                sum_exp += probs[i];
            }
            for (int i = 0; i < vocab_size; ++i) probs[i] /= sum_exp;

            for (int i = 0; i < vocab_size; ++i) top_indices[i] = i;
            std::partial_sort(top_indices.begin(), top_indices.begin() + top_k, top_indices.end(),
                [&probs](int a, int b) { return probs[a] > probs[b]; });

            ss << '[';
            for (int i = 0; i < top_k && i < vocab_size; ++i) {
                if (i > 0) ss << ',';
                int idx = top_indices[i];
                ss << "{\"token\":\"" << json_escape(ctx->tokenizer.id_to_token(idx))
                   << "\",\"prob\":" << probs[idx] << '}';
            }
            ss << ']';
        }
        ss << ']';
    }

    ss << '}';

    return to_c_string(ss.str());
}

char* grasslm_model_config(GrassLMContext handle) {
    if (!handle) return nullptr;

    auto* ctx = static_cast<GrassLMContextImpl*>(handle);
    const grasslm::ModelConfig& cfg = ctx->model.config();

    std::ostringstream ss;
    ss << '{'
       << "\"n_layers\":" << cfg.n_layers
       << ",\"d_model\":" << cfg.d_model
       << ",\"d_reduce\":" << cfg.d_reduce
       << ",\"d_ff\":" << cfg.d_ff
       << ",\"vocab_size\":" << cfg.vocab_size
       << ",\"max_seq_len\":" << cfg.max_seq_len
       << '}';

    return to_c_string(ss.str());
}

}  // extern "C"
