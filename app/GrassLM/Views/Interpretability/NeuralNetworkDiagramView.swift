import SwiftUI

/// Animated architecture diagram showing the GrassLM forward pass.
///
/// Draws a vertical chain of nodes: Input → Embed → Block 0..N-1 → Final LN → LM Head → Output.
/// A pulse dot animates through the pipeline, and each node glows green proportional to its
/// activation norm when the animation reaches it.
struct NeuralNetworkDiagramView: View {
    let debugData: ForwardDebugData
    let archConfig: ModelArchConfig

    @State private var animationProgress: CGFloat = 0
    @State private var hasAppeared = false

    // MARK: - Layout Constants

    private let nodeWidth: CGFloat = 220
    private let blockNodeWidth: CGFloat = 240
    private let nodeHeight: CGFloat = 40
    private let blockNodeHeight: CGFloat = 80
    private let verticalSpacing: CGFloat = 20
    private let connectionLineWidth: CGFloat = 2
    private let dotRadius: CGFloat = 5

    /// Total number of animation stages: embed + N blocks + final LN + LM head.
    private var stageCount: Int {
        archConfig.nLayers + 3 // embed, block×N, finalLN, lmHead
    }

    /// All node descriptors in pipeline order.
    private var nodes: [NodeInfo] {
        var result: [NodeInfo] = []

        result.append(NodeInfo(label: "Tok + Pos Embed", sublabels: nil, norm: meanNorm(debugData.embedNorms)))

        for i in 0..<archConfig.nLayers {
            let blockNorms = i < debugData.blockNorms.count ? debugData.blockNorms[i] : []
            result.append(NodeInfo(
                label: "Block \(i) (w=\(i + 1))",
                sublabels: ["Plücker Encoder", "Gated Fusion + LN", "FFN + LN"],
                norm: meanNorm(blockNorms)
            ))
        }

        // Final layer norm — use the finalNormHeatmap norms
        let finalNorms = debugData.finalNormHeatmap.map { row in
            sqrt(row.reduce(0) { $0 + $1 * $1 })
        }
        result.append(NodeInfo(label: "Final LN", sublabels: nil, norm: meanNorm(finalNorms)))

        result.append(NodeInfo(label: "LM Head", sublabels: nil, norm: nil))

        return result
    }

    /// Normalize norms to [0, 1] range across all nodes for glow intensity.
    private var maxNorm: Float {
        let allNorms = nodes.compactMap(\.norm)
        return allNorms.max() ?? 1
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                sectionHeader("Architecture", icon: "cpu")
                Spacer()
                Button(action: replay) {
                    Label("Replay", systemImage: "arrow.counterclockwise")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.6))
                }
                .buttonStyle(.plain)
            }

            VStack(spacing: 0) {
                // Input label
                inputOutputLabel("Input Tokens", icon: "text.cursor")

                connectionLine()

                // Pipeline nodes
                ForEach(Array(nodes.enumerated()), id: \.offset) { index, node in
                    let stageProgress = stageActivation(for: index)
                    let isBlock = node.sublabels != nil

                    nodeView(node: node, stageProgress: stageProgress, isBlock: isBlock)

                    if index < nodes.count - 1 {
                        connectionWithDot(stageIndex: index)
                    }
                }

                connectionLine()

                // Output label
                inputOutputLabel("Output", icon: "text.badge.checkmark")
            }
            .frame(maxWidth: .infinity)
        }
        .cardBackground()
        .onAppear {
            guard !hasAppeared else { return }
            hasAppeared = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                withAnimation(.linear(duration: 3.0)) {
                    animationProgress = 1.0
                }
            }
        }
    }

    // MARK: - Subviews

    private func inputOutputLabel(_ text: String, icon: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11))
            Text(text)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
        }
        .foregroundStyle(.white.opacity(0.7))
        .frame(maxWidth: .infinity)
    }

    private func nodeView(node: NodeInfo, stageProgress: CGFloat, isBlock: Bool) -> some View {
        let glowIntensity: CGFloat = if let norm = node.norm {
            CGFloat(norm / max(maxNorm, 1e-6)) * stageProgress
        } else {
            stageProgress * 0.5
        }

        let bgColor = Color(
            red: 0.1 * (1 - glowIntensity) + 0.15 * glowIntensity,
            green: 0.1 * (1 - glowIntensity) + 0.55 * glowIntensity,
            blue: 0.1 * (1 - glowIntensity) + 0.2 * glowIntensity
        )

        return VStack(spacing: 4) {
            Text(node.label)
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white)

            if let sublabels = node.sublabels {
                ForEach(sublabels, id: \.self) { sub in
                    Text(sub)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
        }
        .frame(width: isBlock ? blockNodeWidth : nodeWidth,
               height: isBlock ? blockNodeHeight : nodeHeight)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(bgColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(
                    Color.green.opacity(0.1 + 0.5 * glowIntensity),
                    lineWidth: 1 + glowIntensity
                )
        )
        .shadow(color: .green.opacity(0.3 * glowIntensity), radius: 8 * glowIntensity)
    }

    private func connectionLine() -> some View {
        Rectangle()
            .fill(.white.opacity(0.15))
            .frame(width: connectionLineWidth, height: verticalSpacing)
            .frame(maxWidth: .infinity)
    }

    private func connectionWithDot(stageIndex: Int) -> some View {
        let dotProgress = dotPosition(for: stageIndex)

        return ZStack {
            Rectangle()
                .fill(.white.opacity(0.15))
                .frame(width: connectionLineWidth, height: verticalSpacing)

            if dotProgress > 0 && dotProgress < 1 {
                Circle()
                    .fill(.green)
                    .frame(width: dotRadius * 2, height: dotRadius * 2)
                    .shadow(color: .green.opacity(0.8), radius: 6)
                    .offset(y: (dotProgress - 0.5) * verticalSpacing)
            }
        }
        .frame(height: verticalSpacing)
        .frame(maxWidth: .infinity)
    }

    // MARK: - Animation Helpers

    /// Returns 0..1 indicating how "activated" a given stage is.
    private func stageActivation(for index: Int) -> CGFloat {
        let sliceWidth = 1.0 / CGFloat(stageCount)
        let stageStart = CGFloat(index) * sliceWidth
        let stageEnd = stageStart + sliceWidth

        if animationProgress < stageStart { return 0 }
        if animationProgress >= stageEnd { return 1 }
        return (animationProgress - stageStart) / sliceWidth
    }

    /// Returns 0..1 for the pulse dot between stage `index` and `index+1`.
    private func dotPosition(for index: Int) -> CGFloat {
        let sliceWidth = 1.0 / CGFloat(stageCount)
        let transitionStart = CGFloat(index + 1) * sliceWidth - sliceWidth * 0.3
        let transitionEnd = CGFloat(index + 1) * sliceWidth + sliceWidth * 0.3

        if animationProgress < transitionStart { return -1 }
        if animationProgress > transitionEnd { return 2 }
        return (animationProgress - transitionStart) / (transitionEnd - transitionStart)
    }

    private func replay() {
        animationProgress = 0
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            withAnimation(.linear(duration: 3.0)) {
                animationProgress = 1.0
            }
        }
    }

    private func meanNorm(_ norms: [Float]) -> Float {
        guard !norms.isEmpty else { return 0 }
        return norms.reduce(0, +) / Float(norms.count)
    }
}

// MARK: - Node Info

private struct NodeInfo {
    let label: String
    let sublabels: [String]?
    let norm: Float?
}
