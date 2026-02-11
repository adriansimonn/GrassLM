#ifndef GRASSLM_BRIDGE_H
#define GRASSLM_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to the GrassLM engine (model + tokenizer + generator).
typedef void* GrassLMContext;

/// Create a GrassLM context by loading model weights and vocabulary.
/// Returns NULL on failure.
GrassLMContext grasslm_create(const char* model_path, const char* vocab_path);

/// Destroy a GrassLM context and free all resources.
void grasslm_destroy(GrassLMContext ctx);

/// Generate text from a prompt (blocking, returns full result).
/// Caller must free the returned string with grasslm_free_string().
/// Returns NULL on failure.
char* grasslm_generate(GrassLMContext ctx, const char* prompt, int max_tokens,
                        float temperature, float top_p);

/// Callback invoked for each generated token during streaming.
/// token: the decoded token string (valid only for the duration of the call).
/// user_data: opaque pointer passed through from grasslm_generate_stream.
typedef void (*TokenCallback)(const char* token, void* user_data);

/// Generate text from a prompt with streaming token-by-token output.
/// Calls the callback for each generated token.
void grasslm_generate_stream(GrassLMContext ctx, const char* prompt,
                              int max_tokens, float temperature, float top_p,
                              TokenCallback callback, void* user_data);

/// Free a string returned by grasslm_generate().
void grasslm_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif /* GRASSLM_BRIDGE_H */
