import SwiftUI

/// A message row styled like Claude/ChatGPT:
/// - User messages: right-aligned in a rounded rectangle
/// - Assistant messages: left-aligned, plain text on the page
struct MessageBubble: View {
    let message: PersistableMessage
    var onTap: (() -> Void)? = nil

    private var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if isUser {
                Spacer(minLength: 80)
                userBubble
            } else {
                assistantView
                Spacer(minLength: 80)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 6)
    }

    // MARK: - User Message

    private var userBubble: some View {
        Text(message.content)
            .textSelection(.enabled)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color.accentColor.opacity(0.15))
            .foregroundStyle(.primary)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .contentShape(RoundedRectangle(cornerRadius: 20))
            .onTapGesture { onTap?() }
    }

    // MARK: - Assistant Message

    private var assistantView: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "leaf.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
                Text(modelDisplayName)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
            }

            Text(message.content)
                .textSelection(.enabled)
                .lineSpacing(3)
        }
        .contentShape(Rectangle())
        .onTapGesture { onTap?() }
    }

    private var modelDisplayName: String {
        if let id = message.modelID,
           let model = ModelInfo.available.first(where: { $0.id == id }) {
            return model.displayName
        }
        return "GrassLM"
    }
}
