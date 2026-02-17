import SwiftUI

/// Full-screen overlay that displays interpretability visuals for a tapped message.
///
/// - **User message**: TokenizationView → EmbeddingHeatmapView
/// - **Assistant message**: NeuralNetworkDiagramView → LayerActivationView → TokenizationView → TopPredictionsView
struct InterpretabilityOverlay: View {
    let payload: InterpretabilityPayload
    let onDismiss: () -> Void

    @State private var isVisible = false

    private var title: String {
        switch payload {
        case .userMessage: "Input Analysis"
        case .assistantMessage: "Output Analysis"
        }
    }

    var body: some View {
        ZStack {
            // Blurred dark background
            Rectangle()
                .fill(.ultraThinMaterial)
                .environment(\.colorScheme, .dark)
                .overlay(Color.black.opacity(0.5))
                .ignoresSafeArea()
                .onTapGesture { dismiss() }

            // Content
            VStack(spacing: 0) {
                topBar
                    .padding(.horizontal, 24)
                    .padding(.top, 48)
                    .padding(.bottom, 8)

                ScrollView {
                    VStack(spacing: 20) {
                        switch payload {
                        case .userMessage(let tokenization, let debugData):
                            userMessageContent(tokenization: tokenization, debugData: debugData)
                        case .assistantMessage(let tokenization, let debugData, let archConfig):
                            assistantMessageContent(tokenization: tokenization, debugData: debugData, archConfig: archConfig)
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.bottom, 32)
                    .frame(maxWidth: 900)
                    .frame(maxWidth: .infinity)
                }
            }
        }
        .opacity(isVisible ? 1 : 0)
        .scaleEffect(isVisible ? 1.0 : 0.97)
        .onAppear {
            withAnimation(.easeOut(duration: 0.25)) {
                isVisible = true
            }
        }
    }

    // MARK: - Top Bar

    private var topBar: some View {
        ZStack {
            // Center title
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)

            // Back button on left
            HStack {
                Button(action: dismiss) {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 14, weight: .semibold))
                        Text("Back")
                            .font(.system(size: 14, weight: .medium))
                    }
                    .foregroundStyle(.white.opacity(0.7))
                }
                .buttonStyle(.plain)
                .onHover { inside in
                    if inside { NSCursor.pointingHand.push() } else { NSCursor.pop() }
                }

                Spacer()
            }
        }
    }

    // MARK: - User Message Content

    @ViewBuilder
    private func userMessageContent(tokenization: TokenizationResult, debugData: ForwardDebugData) -> some View {
        TokenizationView(tokenization: tokenization)

        EmbeddingHeatmapView(
            tokens: tokenization.tokens,
            heatmap: debugData.embedHeatmap,
            title: "Embedding Activations"
        )
    }

    // MARK: - Assistant Message Content

    @ViewBuilder
    private func assistantMessageContent(tokenization: TokenizationResult, debugData: ForwardDebugData, archConfig: ModelArchConfig) -> some View {
        NeuralNetworkDiagramView(debugData: debugData, archConfig: archConfig, tokens: tokenization.tokens)

        LayerActivationView(tokens: tokenization.tokens, debugData: debugData)

        TokenizationView(tokenization: tokenization)
    }

    // MARK: - Dismiss

    private func dismiss() {
        withAnimation(.easeIn(duration: 0.2)) {
            isVisible = false
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            onDismiss()
        }
    }
}
