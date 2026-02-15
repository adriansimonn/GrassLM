import Foundation

/// Manages local persistence of conversations as JSON files in Application Support.
final class ChatStore {
    static let shared = ChatStore()

    private let fileManager = FileManager.default
    private let storageDirectory: URL

    private init() {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        storageDirectory = appSupport.appendingPathComponent("GrassLM/conversations", isDirectory: true)

        // Ensure directory exists
        try? fileManager.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
    }

    // MARK: - File Helpers

    private func fileURL(for conversationID: UUID) -> URL {
        storageDirectory.appendingPathComponent("\(conversationID.uuidString).json")
    }

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = .prettyPrinted
        return e
    }()

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    // MARK: - CRUD

    /// Load all conversations, sorted by most recently updated.
    func loadAll() -> [Conversation] {
        guard let files = try? fileManager.contentsOfDirectory(at: storageDirectory, includingPropertiesForKeys: nil) else {
            return []
        }

        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> Conversation? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(Conversation.self, from: data)
            }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    /// Save a single conversation to disk.
    func save(_ conversation: Conversation) {
        let url = fileURL(for: conversation.id)
        guard let data = try? encoder.encode(conversation) else { return }
        try? data.write(to: url, options: .atomic)
    }

    /// Delete a conversation from disk.
    func delete(_ conversationID: UUID) {
        let url = fileURL(for: conversationID)
        try? fileManager.removeItem(at: url)
    }
}
