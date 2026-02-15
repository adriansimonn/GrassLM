import Foundation

/// A persistable message within a conversation.
struct PersistableMessage: Identifiable, Codable {
    let id: UUID
    let role: MessageRole
    var content: String
    let timestamp: Date

    enum MessageRole: String, Codable {
        case user
        case assistant
    }

    init(role: MessageRole, content: String) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.timestamp = Date()
    }
}

/// A conversation containing a list of messages, persistable to disk.
struct Conversation: Identifiable, Codable {
    let id: UUID
    var title: String
    var messages: [PersistableMessage]
    let createdAt: Date
    var updatedAt: Date

    init(title: String = "New Chat") {
        self.id = UUID()
        self.title = title
        self.messages = []
        self.createdAt = Date()
        self.updatedAt = Date()
    }

    /// Derive a title from the first user message, truncated.
    mutating func updateTitleFromFirstMessage() {
        guard let first = messages.first(where: { $0.role == .user }) else { return }
        let text = first.content.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.count > 40 {
            title = String(text.prefix(40)) + "..."
        } else {
            title = text
        }
    }
}
