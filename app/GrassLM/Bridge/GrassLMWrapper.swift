import Foundation

/// Errors that can occur when using the GrassLM engine.
enum GrassLMError: Error, LocalizedError {
    case failedToLoad(modelPath: String, vocabPath: String)
    case generationFailed

    var errorDescription: String? {
        switch self {
        case .failedToLoad(let modelPath, let vocabPath):
            return "Failed to load GrassLM model from \(modelPath) with vocab \(vocabPath)"
        case .generationFailed:
            return "Text generation failed"
        }
    }
}

/// Swift wrapper around the GrassLM C++ inference engine.
///
/// Manages model lifecycle and provides async/await text generation
/// with streaming support via `AsyncStream<String>`.
final class GrassLMWrapper: @unchecked Sendable {
    /// Opaque handle to the C engine context.
    private var context: GrassLMContext?

    /// Serial queue for all inference operations (C++ engine is not thread-safe).
    private let queue = DispatchQueue(label: "com.grasslm.inference", qos: .userInitiated)

    /// Whether the model is loaded and ready for generation.
    var isLoaded: Bool { context != nil }

    // MARK: - Lifecycle

    /// Initialize by loading a model and vocabulary from disk.
    /// - Parameters:
    ///   - modelPath: Path to the `.grasslm` binary weights file.
    ///   - vocabPath: Path to the `vocab.txt` WordPiece vocabulary file.
    /// - Throws: `GrassLMError.failedToLoad` if loading fails.
    init(modelPath: String, vocabPath: String) throws {
        guard let ctx = grasslm_create(modelPath, vocabPath) else {
            throw GrassLMError.failedToLoad(modelPath: modelPath, vocabPath: vocabPath)
        }
        self.context = ctx
    }

    deinit {
        if let ctx = context {
            grasslm_destroy(ctx)
        }
    }

    // MARK: - Generation (blocking, returns full result)

    /// Generate text from a prompt. Runs on a background queue.
    /// - Parameters:
    ///   - prompt: The input text to continue from.
    ///   - maxTokens: Maximum number of tokens to generate.
    ///   - temperature: Sampling temperature (0 = greedy, higher = more random).
    ///   - topP: Nucleus sampling threshold (1.0 = disabled).
    /// - Returns: The generated text.
    /// - Throws: `GrassLMError.generationFailed` if generation fails.
    func generate(
        prompt: String,
        maxTokens: Int = 128,
        temperature: Float = 0.8,
        topP: Float = 0.9
    ) async throws -> String {
        guard let ctx = context else {
            throw GrassLMError.generationFailed
        }

        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                guard let cStr = grasslm_generate(
                    ctx, prompt,
                    Int32(maxTokens),
                    temperature,
                    topP
                ) else {
                    continuation.resume(throwing: GrassLMError.generationFailed)
                    return
                }

                let result = String(cString: cStr)
                grasslm_free_string(cStr)
                continuation.resume(returning: result)
            }
        }
    }

    // MARK: - Streaming generation

    /// Generate text token-by-token, returning an `AsyncStream` that yields each token as it is produced.
    /// - Parameters:
    ///   - prompt: The input text to continue from.
    ///   - maxTokens: Maximum number of tokens to generate.
    ///   - temperature: Sampling temperature (0 = greedy, higher = more random).
    ///   - topP: Nucleus sampling threshold (1.0 = disabled).
    /// - Returns: An `AsyncStream<String>` that yields individual tokens.
    func generateStream(
        prompt: String,
        maxTokens: Int = 128,
        temperature: Float = 0.8,
        topP: Float = 0.9
    ) -> AsyncStream<String> {
        guard let ctx = context else {
            return AsyncStream { $0.finish() }
        }

        return AsyncStream { continuation in
            queue.async {
                // Bridge context passed through the C callback's user_data pointer.
                // We use an Unmanaged reference to the continuation to avoid ARC issues
                // across the C function boundary.
                typealias ContinuationType = AsyncStream<String>.Continuation

                let callbackContext = UnsafeMutablePointer<ContinuationType>.allocate(capacity: 1)
                callbackContext.initialize(to: continuation)

                let callback: TokenCallback = { tokenCStr, userData in
                    guard let tokenCStr = tokenCStr, let userData = userData else { return }
                    let cont = userData.assumingMemoryBound(to: ContinuationType.self).pointee
                    let token = String(cString: tokenCStr)
                    cont.yield(token)
                }

                grasslm_generate_stream(
                    ctx, prompt,
                    Int32(maxTokens),
                    temperature,
                    topP,
                    callback,
                    callbackContext
                )

                // Generation complete — clean up and finish the stream
                callbackContext.deinitialize(count: 1)
                callbackContext.deallocate()
                continuation.finish()
            }
        }
    }
}
