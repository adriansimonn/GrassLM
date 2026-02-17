import SwiftUI

/// Chat history sidebar with a liquid glass translucent appearance.
struct SidebarView: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var hoveredID: UUID?
    @State private var renamingID: UUID?
    @State private var renameText: String = ""

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Generations")
                    .font(.headline)
                    .foregroundStyle(.primary)
                Spacer()
                Button {
                    viewModel.createNewConversation()
                } label: {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 14, weight: .medium))
                }
                .buttonStyle(.plain)
                .help("New Chat")
            }
            .padding(.horizontal, 16)
            .padding(.top, 38)
            .padding(.bottom, 12)

            Divider()
                .opacity(0.5)

            // Conversation list
            if viewModel.conversations.isEmpty {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "bubble.left.and.bubble.right")
                        .font(.title2)
                        .foregroundStyle(.tertiary)
                    Text("No conversations yet")
                        .font(.subheadline)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 2) {
                        ForEach(viewModel.conversations) { conversation in
                            conversationRow(conversation)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 8)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(sidebarBackground)
    }

    // MARK: - Conversation Row

    private func conversationRow(_ conversation: Conversation) -> some View {
        let isSelected = viewModel.currentConversation?.id == conversation.id
        let isHovered = hoveredID == conversation.id
        let isRenaming = renamingID == conversation.id

        return Button {
            if !isRenaming {
                viewModel.selectConversation(conversation.id)
            }
        } label: {
            VStack(alignment: .leading, spacing: 3) {
                if isRenaming {
                    TextField("Chat name", text: $renameText, onCommit: {
                        viewModel.renameConversation(conversation.id, to: renameText)
                        renamingID = nil
                    })
                    .font(.subheadline)
                    .textFieldStyle(.plain)
                } else {
                    Text(conversation.title)
                        .font(.subheadline)
                        .fontWeight(isSelected ? .semibold : .regular)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .foregroundStyle(isSelected ? .primary : .secondary)
                }

                Text(conversation.updatedAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isSelected
                          ? Color.accentColor.opacity(0.15)
                          : isHovered ? Color.primary.opacity(0.05) : Color.clear)
            )
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            hoveredID = hovering ? conversation.id : nil
        }
        .contextMenu {
            Button {
                renameText = conversation.title
                renamingID = conversation.id
            } label: {
                Label("Rename", systemImage: "pencil")
            }

            Button(role: .destructive) {
                viewModel.deleteConversation(conversation.id)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    // MARK: - Liquid Glass Background

    private var sidebarBackground: some View {
        ZStack {
            // Base translucent layer — hudWindow is more transparent than sidebar
            VisualEffectBlur(material: .hudWindow, blendingMode: .behindWindow)

            // Subtle gradient overlay for the glass effect
            LinearGradient(
                colors: [
                    Color.white.opacity(0.03),
                    Color.clear,
                    Color.white.opacity(0.015)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            // Inner border highlight
            RoundedRectangle(cornerRadius: 0)
                .strokeBorder(
                    LinearGradient(
                        colors: [
                            Color.white.opacity(0.08),
                            Color.white.opacity(0.02)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    lineWidth: 0.5
                )
        }
    }
}

// MARK: - NSVisualEffectView Wrapper

/// Wraps NSVisualEffectView for macOS translucent/blur backgrounds.
struct VisualEffectBlur: NSViewRepresentable {
    var material: NSVisualEffectView.Material
    var blendingMode: NSVisualEffectView.BlendingMode

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
    }
}
