import SwiftUI

/// Renders a 2D activation heatmap using SwiftUI Canvas.
///
/// Rows correspond to token positions, columns to embedding dimensions.
/// Token labels are shown along the left side. Uses the shared `heatmapColor` mapping.
struct EmbeddingHeatmapView: View {
    let tokens: [String]
    let heatmap: [[Float]]
    let title: String

    /// Maximum columns to render. Downsamples if the heatmap is wider.
    private let maxDisplayCols = 256

    /// Label area width to the left of the heatmap grid.
    private let labelWidth: CGFloat = 70

    /// Height of each token row.
    private let rowHeight: CGFloat = 22

    /// Normalized heatmap values in [0, 1], downsampled if needed.
    private var normalizedData: [[Float]] {
        guard let first = heatmap.first, !first.isEmpty else { return [] }

        let allValues = heatmap.flatMap { $0 }
        let minVal = allValues.min() ?? 0
        let maxVal = allValues.max() ?? 1
        let range = maxVal - minVal
        let safeRange = range > 1e-8 ? range : 1

        return heatmap.map { row in
            let downsampled = downsample(row, to: maxDisplayCols)
            return downsampled.map { ($0 - minVal) / safeRange }
        }
    }

    private var displayCols: Int {
        guard let first = heatmap.first else { return 0 }
        return min(first.count, maxDisplayCols)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader(title, icon: "square.grid.3x3.fill")

            let rows = normalizedData.count
            let cols = displayCols
            let gridWidth: CGFloat = max(CGFloat(cols) * 2.5, 200)
            let totalHeight = CGFloat(rows) * rowHeight

            ScrollView(.horizontal, showsIndicators: true) {
                HStack(alignment: .top, spacing: 0) {
                    // Token labels
                    VStack(alignment: .trailing, spacing: 0) {
                        ForEach(0..<rows, id: \.self) { row in
                            let label = row < tokens.count ? tokens[row] : "[\(row)]"
                            Text(truncateToken(label, maxLen: 8))
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.6))
                                .frame(width: labelWidth, height: rowHeight, alignment: .trailing)
                                .padding(.trailing, 4)
                        }
                    }

                    // Canvas heatmap
                    Canvas { context, size in
                        let cellWidth = cols > 0 ? size.width / CGFloat(cols) : 0
                        let cellHeight = rows > 0 ? size.height / CGFloat(rows) : 0

                        for row in 0..<normalizedData.count {
                            for col in 0..<normalizedData[row].count {
                                let color = heatmapColor(normalizedData[row][col])
                                let rect = CGRect(
                                    x: CGFloat(col) * cellWidth,
                                    y: CGFloat(row) * cellHeight,
                                    width: cellWidth + 0.5, // slight overlap to avoid gaps
                                    height: cellHeight + 0.5
                                )
                                context.fill(Path(rect), with: .color(color))
                            }
                        }
                    }
                    .frame(width: gridWidth, height: totalHeight)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                }
            }

            // Dimension info
            HStack(spacing: 16) {
                Text("\(rows) tokens \u{00d7} \(heatmap.first?.count ?? 0) dims")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.35))

                // Color legend
                HStack(spacing: 4) {
                    Text("low")
                        .font(.system(size: 10))
                        .foregroundStyle(.white.opacity(0.35))
                    legendGradient
                        .frame(width: 60, height: 8)
                        .clipShape(RoundedRectangle(cornerRadius: 2))
                    Text("high")
                        .font(.system(size: 10))
                        .foregroundStyle(.white.opacity(0.35))
                }
            }
        }
        .cardBackground()
    }

    /// Gradient bar showing the heatmap color scale.
    private var legendGradient: some View {
        Canvas { context, size in
            let steps = 32
            let stepWidth = size.width / CGFloat(steps)
            for i in 0..<steps {
                let t = Float(i) / Float(steps - 1)
                let rect = CGRect(
                    x: CGFloat(i) * stepWidth,
                    y: 0,
                    width: stepWidth + 0.5,
                    height: size.height
                )
                context.fill(Path(rect), with: .color(heatmapColor(t)))
            }
        }
    }

    /// Downsample a row by averaging groups of values.
    private func downsample(_ row: [Float], to maxCols: Int) -> [Float] {
        guard row.count > maxCols else { return row }
        let groupSize = Float(row.count) / Float(maxCols)
        return (0..<maxCols).map { i in
            let start = Int(Float(i) * groupSize)
            let end = min(Int(Float(i + 1) * groupSize), row.count)
            guard end > start else { return 0 }
            let sum = row[start..<end].reduce(0, +)
            return sum / Float(end - start)
        }
    }

    private func truncateToken(_ token: String, maxLen: Int) -> String {
        if token.count <= maxLen { return token }
        return String(token.prefix(maxLen - 1)) + "\u{2026}"
    }
}
