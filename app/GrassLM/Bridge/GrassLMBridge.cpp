#include "GrassLMBridge.h"

#include <grasslm/generator.h>
#include <grasslm/model.h>
#include <grasslm/tokenizer.h>
#include <grasslm/weight_loader.h>

#include <cstdlib>
#include <cstring>
#include <string>

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

}  // extern "C"
