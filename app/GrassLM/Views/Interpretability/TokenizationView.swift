import SwiftUI

// MARK: - Token Pill

/// A single token displayed as a pill with subword text and token ID.
struct TokenPill: View {
    let token: String
    let tokenID: Int

    @State private var isHovered = false

    /// WordPiece continuation tokens (e.g. "##ing") get distinct orange styling.
    private var isContinuation: Bool {
        token.hasPrefix("##")
    }

    /// Display text strips the "##" prefix for continuation tokens.
    private var displayText: String {
        isContinuation ? String(token.dropFirst(2)) : token
    }

    private var pillColor: Color {
        if isContinuation {
            return Color.orange.opacity(isHovered ? 0.35 : 0.2)
        }
        return Color.white.opacity(isHovered ? 0.15 : 0.08)
    }

    private var borderColor: Color {
        if isContinuation {
            return Color.orange.opacity(isHovered ? 0.5 : 0.3)
        }
        return Color.white.opacity(isHovered ? 0.2 : 0.1)
    }

    var body: some View {
        VStack(spacing: 2) {
            Text(displayText)
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundStyle(isContinuation ? .orange : .white)

            Text("\(tokenID)")
                .font(.system(size: 9, weight: .regular, design: .monospaced))
                .foregroundStyle(.white.opacity(0.4))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(pillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(borderColor, lineWidth: 1)
        )
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
    }
}

// MARK: - Tokenization View

/// Displays all tokens from a tokenization result as a flowing grid of pills.
struct TokenizationView: View {
    let tokenization: TokenizationResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Tokens", icon: "textformat.abc")

            FlowLayout(spacing: 6) {
                ForEach(Array(zip(tokenization.tokens, tokenization.ids).enumerated()), id: \.offset) { _, pair in
                    TokenPill(token: pair.0, tokenID: pair.1)
                }
            }

            Text("\(tokenization.tokens.count) tokens")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(0.4))
        }
        .cardBackground()
    }
}
