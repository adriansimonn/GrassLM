import Foundation
import SwiftUI

/// A single message in the conversation.
struct Message: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    let timestamp: Date

    enum Role {
        case user
        case assistant
    }

    init(role: Role, content: String) {
        self.role = role
        self.content = content
        self.timestamp = Date()
    }
}

/// Observable ViewModel managing conversation state and generation.
@MainActor
final class ChatViewModel: ObservableObject {
    // MARK: - Published State

    @Published var messages: [Message] = []
    @Published var inputText: String = ""
    @Published var isGenerating: Bool = false
    @Published var isModelLoaded: Bool = false
    @Published var isLoadingModel: Bool = false
    @Published var errorMessage: String?

    // MARK: - Generation Settings

    @Published var temperature: Float = 0.8
    @Published var topP: Float = 0.9
    @Published var maxTokens: Int = 128

    // MARK: - Private

    private var engine: GrassLMWrapper?
    private var generationTask: Task<Void, Never>?

    // MARK: - Model Loading

    /// Load the GrassLM model from bundled resources.
    func loadModel() {
        guard !isModelLoaded && !isLoadingModel else { return }
        isLoadingModel = true
        errorMessage = nil

        Task {
            do {
                let modelPath = Self.bundledModelPath()
                let vocabPath = Self.bundledVocabPath()

                let wrapper = try await Task.detached(priority: .userInitiated) {
                    try GrassLMWrapper(modelPath: modelPath, vocabPath: vocabPath)
                }.value

                self.engine = wrapper
                self.isModelLoaded = true
            } catch {
                self.errorMessage = error.localizedDescription
            }
            self.isLoadingModel = false
        }
    }

    // MARK: - Generation

    /// Send the current input text and generate an assistant response.
    func send() {
        let prompt = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isGenerating, isModelLoaded else { return }

        // Append user message
        messages.append(Message(role: .user, content: prompt))
        inputText = ""

        // Start generation
        generate(prompt: prompt)
    }

    /// Cancel any in-flight generation.
    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    // MARK: - Private Helpers

    private func generate(prompt: String) {
        guard let engine = engine else { return }

        isGenerating = true
        errorMessage = nil

        // Append a placeholder assistant message that we'll stream into
        let assistantMessage = Message(role: .assistant, content: "")
        messages.append(assistantMessage)
        let messageIndex = messages.count - 1

        generationTask = Task {
            let stream = engine.generateStream(
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP
            )

            for await token in stream {
                if Task.isCancelled { break }
                messages[messageIndex].content += token
            }

            // Remove empty assistant messages (e.g., if generation failed immediately)
            if messages[messageIndex].content.isEmpty {
                messages.remove(at: messageIndex)
            }

            isGenerating = false
            generationTask = nil
        }
    }

    // MARK: - Resource Paths

    private static func bundledModelPath() -> String {
        if let path = Bundle.main.path(forResource: "grasslm-6L", ofType: "grasslm") {
            return path
        }
        // Fallback for development: look in the project Resources directory
        return "grasslm-6L.grasslm"
    }

    private static func bundledVocabPath() -> String {
        if let path = Bundle.main.path(forResource: "vocab", ofType: "txt") {
            return path
        }
        return "vocab.txt"
    }
}
