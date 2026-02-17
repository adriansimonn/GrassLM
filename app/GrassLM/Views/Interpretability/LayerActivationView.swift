import SwiftUI

/// Per-layer activation browser with a pill selector strip and norm bar chart.
///
/// Lets users switch between "Embed", "Block 0" ... "Block N-1", and "Final LN",
/// showing the corresponding activation heatmap and L2 norm bar chart.
struct LayerActivationView: View {
    let tokens: [String]
    let debugData: ForwardDebugData

    @State private var selectedLayer: Int = 0

    /// Labels for the layer selector pills.
    private var layerLabels: [String] {
        var labels = ["Embed"]
        for i in 0..<debugData.nLayers {
            labels.append("Block \(i)")
        }
        labels.append("Final LN")
        return labels
    }

    /// Heatmap data for the currently selected layer.
    private var selectedHeatmap: [[Float]] {
        if selectedLayer == 0 {
            return debugData.embedHeatmap
        } else if selectedLayer <= debugData.nLayers {
            let blockIndex = selectedLayer - 1
            if blockIndex < debugData.blockHeatmaps.count {
                return debugData.blockHeatmaps[blockIndex]
            }
            return []
        } else {
            return debugData.finalNormHeatmap
        }
    }

    /// L2 norms per token position for the selected layer.
    private var selectedNorms: [Float] {
        if selectedLayer == 0 {
            return debugData.embedNorms
        } else if selectedLayer <= debugData.nLayers {
            let blockIndex = selectedLayer - 1
            if blockIndex < debugData.blockNorms.count {
                return debugData.blockNorms[blockIndex]
            }
            return []
        } else {
            // Compute norms from finalNormHeatmap rows
            return debugData.finalNormHeatmap.map { row in
                sqrt(row.reduce(0) { $0 + $1 * $1 })
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            sectionHeader("Layer Activations", icon: "chart.bar.xaxis")

            // Layer selector strip
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Array(layerLabels.enumerated()), id: \.offset) { index, label in
                        layerPill(label: label, index: index)
                    }
                }
                .padding(.horizontal, 2)
            }

            // Heatmap for selected layer
            if !selectedHeatmap.isEmpty {
                EmbeddingHeatmapView(
                    tokens: tokens,
                    heatmap: selectedHeatmap,
                    title: "\(layerLabels[selectedLayer]) Activations"
                )
            }

            // Norm bar chart
            if !selectedNorms.isEmpty {
                NormBarChart(tokens: tokens, norms: selectedNorms)
            }
        }
        .cardBackground()
    }

    private func layerPill(label: String, index: Int) -> some View {
        let isSelected = selectedLayer == index

        return Button {
            withAnimation(.easeInOut(duration: 0.2)) {
                selectedLayer = index
            }
        } label: {
            Text(label)
                .font(.system(size: 12, weight: isSelected ? .semibold : .regular, design: .monospaced))
                .foregroundStyle(isSelected ? .white : .white.opacity(0.5))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isSelected ? .green.opacity(0.25) : .white.opacity(0.06))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(isSelected ? .green.opacity(0.4) : .clear, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Norm Bar Chart

/// Horizontal bar chart showing L2 norm per token position.
struct NormBarChart: View {
    let tokens: [String]
    let norms: [Float]

    private var maxNorm: Float {
        norms.max() ?? 1
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("L2 Norms")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.white.opacity(0.6))

            VStack(spacing: 4) {
                ForEach(Array(norms.enumerated()), id: \.offset) { index, norm in
                    HStack(spacing: 8) {
                        let label = index < tokens.count ? tokens[index] : "[\(index)]"
                        Text(truncateToken(label, maxLen: 6))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.5))
                            .frame(width: 50, alignment: .trailing)

                        GeometryReader { geo in
                            let barWidth = maxNorm > 0
                                ? CGFloat(norm / maxNorm) * geo.size.width
                                : 0

                            RoundedRectangle(cornerRadius: 3)
                                .fill(barColor(fraction: norm / max(maxNorm, 1e-6)))
                                .frame(width: max(barWidth, 2), height: geo.size.height)
                        }
                        .frame(height: 14)

                        Text(String(format: "%.1f", norm))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.4))
                            .frame(width: 40, alignment: .leading)
                    }
                }
            }
        }
    }

    private func barColor(fraction: Float) -> Color {
        let t = min(max(fraction, 0), 1)
        return Color(
            red: Double(t) * 0.3,
            green: 0.4 + Double(t) * 0.4,
            blue: 0.3 * Double(1 - t)
        )
    }

    private func truncateToken(_ token: String, maxLen: Int) -> String {
        if token.count <= maxLen { return token }
        return String(token.prefix(maxLen - 1)) + "\u{2026}"
    }
}
