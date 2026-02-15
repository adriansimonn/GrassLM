import Foundation
import SwiftUI

/// Observable ViewModel managing conversations, generation, and model selection.
@MainActor
final class ChatViewModel: ObservableObject {
    // MARK: - Published State

    @Published var conversations: [Conversation] = []
    @Published var currentConversation: Conversation?
    @Published var inputText: String = ""
    @Published var isGenerating: Bool = false
    @Published var isModelLoaded: Bool = false
    @Published var isLoadingModel: Bool = false
    @Published var errorMessage: String?

    // MARK: - Model Selection

    @Published var selectedModelID: String = ModelInfo.available.first?.id ?? ""

    // MARK: - Generation Settings (persisted via UserDefaults)

    @Published var temperature: Float = 0.8 {
        didSet { defaults.set(temperature, forKey: "gen_temperature") }
    }
    @Published var topP: Float = 0.9 {
        didSet { defaults.set(topP, forKey: "gen_topP") }
    }
    @Published var maxTokens: Int = 128 {
        didSet { defaults.set(maxTokens, forKey: "gen_maxTokens") }
    }

    // MARK: - Private

    private var engine: GrassLMWrapper?
    private var generationTask: Task<Void, Never>?
    private let store = ChatStore.shared
    private let defaults = UserDefaults.standard

    // MARK: - Computed

    /// Messages for the current conversation.
    var messages: [PersistableMessage] {
        currentConversation?.messages ?? []
    }

    // MARK: - Initialization

    init() {
        conversations = store.loadAll()

        // Restore persisted generation settings
        if defaults.object(forKey: "gen_temperature") != nil {
            temperature = defaults.float(forKey: "gen_temperature")
        }
        if defaults.object(forKey: "gen_topP") != nil {
            topP = defaults.float(forKey: "gen_topP")
        }
        if defaults.object(forKey: "gen_maxTokens") != nil {
            maxTokens = defaults.integer(forKey: "gen_maxTokens")
        }
    }

    // MARK: - Model Loading

    /// Load the GrassLM model from bundled resources.
    func loadModel() {
        guard !isModelLoaded && !isLoadingModel else { return }
        isLoadingModel = true
        errorMessage = nil

        Task {
            do {
                let modelPath = Self.bundledModelPath(for: selectedModelID)
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

    // MARK: - Conversation Management

    /// Create a new conversation and make it active.
    func createNewConversation() {
        // If the current conversation is empty, just keep it
        if let current = currentConversation, current.messages.isEmpty {
            return
        }

        let conversation = Conversation()
        store.save(conversation)
        conversations.insert(conversation, at: 0)
        currentConversation = conversation
    }

    /// Select an existing conversation by ID.
    func selectConversation(_ id: UUID) {
        guard let conversation = conversations.first(where: { $0.id == id }) else { return }
        cancelGeneration()
        currentConversation = conversation
    }

    /// Rename a conversation by ID.
    func renameConversation(_ id: UUID, to newTitle: String) {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if let index = conversations.firstIndex(where: { $0.id == id }) {
            conversations[index].title = trimmed
            conversations[index].updatedAt = Date()
            store.save(conversations[index])

            if currentConversation?.id == id {
                currentConversation = conversations[index]
            }
        }
    }

    /// Delete a conversation by ID.
    func deleteConversation(_ id: UUID) {
        store.delete(id)
        conversations.removeAll { $0.id == id }

        if currentConversation?.id == id {
            currentConversation = conversations.first
        }
    }

    /// Persist the current conversation to disk.
    private func saveCurrentConversation() {
        guard var conversation = currentConversation else { return }
        conversation.updatedAt = Date()
        store.save(conversation)

        // Update in-memory list
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = conversation
        }

        // Re-sort so most recent is first
        conversations.sort { $0.updatedAt > $1.updatedAt }
        currentConversation = conversation
    }

    // MARK: - Generation

    /// Send the current input text and generate an assistant response.
    func send() {
        let prompt = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isGenerating, isModelLoaded else { return }

        // Create a conversation if none exists
        if currentConversation == nil {
            createNewConversation()
        }

        // Append user message
        let userMessage = PersistableMessage(role: .user, content: prompt)
        currentConversation?.messages.append(userMessage)

        // Auto-title from first message
        if currentConversation?.messages.count == 1 {
            currentConversation?.updateTitleFromFirstMessage()
        }

        inputText = ""
        saveCurrentConversation()

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
        let assistantMessage = PersistableMessage(role: .assistant, content: "")
        currentConversation?.messages.append(assistantMessage)
        let messageIndex = (currentConversation?.messages.count ?? 1) - 1

        generationTask = Task {
            let stream = engine.generateStream(
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP
            )

            for await token in stream {
                if Task.isCancelled { break }
                currentConversation?.messages[messageIndex].content += token
            }

            // Remove empty assistant messages (e.g., if generation failed immediately)
            if let content = currentConversation?.messages[messageIndex].content, content.isEmpty {
                currentConversation?.messages.remove(at: messageIndex)
            }

            saveCurrentConversation()
            isGenerating = false
            generationTask = nil
        }
    }

    // MARK: - Resource Paths

    private static func bundledModelPath(for modelID: String) -> String {
        let model = ModelInfo.available.first(where: { $0.id == modelID }) ?? ModelInfo.available[0]
        if let path = Bundle.main.path(forResource: model.resourceName, ofType: model.fileExtension) {
            return path
        }
        return "\(model.resourceName).\(model.fileExtension)"
    }

    private static func bundledVocabPath() -> String {
        if let path = Bundle.main.path(forResource: "vocab", ofType: "txt") {
            return path
        }
        return "vocab.txt"
    }
}
