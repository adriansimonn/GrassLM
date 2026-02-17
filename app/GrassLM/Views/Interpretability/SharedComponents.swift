import SwiftUI

// MARK: - Section Header

/// Styled section header with an SF Symbol icon and white title text.
func sectionHeader(_ title: String, icon: String) -> some View {
    HStack(spacing: 8) {
        Image(systemName: icon)
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(.white.opacity(0.8))
        Text(title)
            .font(.headline)
            .foregroundStyle(.white)
    }
}

// MARK: - Card Background

/// Dark translucent rounded rect with a subtle border, used as a container background.
struct CardBackground: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(.white.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 1)
            )
    }
}

extension View {
    func cardBackground() -> some View {
        modifier(CardBackground())
    }
}

// MARK: - Heatmap Color

/// Maps a normalized value `t` in [0, 1] to a blue → dark → green → yellow colormap.
func heatmapColor(_ t: Float) -> Color {
    let clamped = min(max(t, 0), 1)

    // 0.0 = blue, 0.33 = dark, 0.66 = green, 1.0 = yellow
    let r: Double
    let g: Double
    let b: Double

    if clamped < 0.33 {
        // Blue → dark
        let p = Double(clamped / 0.33)
        r = 0
        g = 0
        b = 0.6 * (1 - p)
    } else if clamped < 0.66 {
        // Dark → green
        let p = Double((clamped - 0.33) / 0.33)
        r = 0
        g = 0.7 * p
        b = 0
    } else {
        // Green → yellow
        let p = Double((clamped - 0.66) / 0.34)
        r = 0.9 * p
        g = 0.7 + 0.3 * p
        b = 0
    }

    return Color(red: r, green: g, blue: b)
}

// MARK: - Flow Layout

/// A wrapping horizontal layout that flows items to the next row when they exceed the available width.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        var height: CGFloat = 0
        for (index, row) in rows.enumerated() {
            let rowHeight = row.map { subviews[$0].sizeThatFits(.unspecified).height }.max() ?? 0
            height += rowHeight
            if index < rows.count - 1 {
                height += spacing
            }
        }
        return CGSize(width: proposal.width ?? 0, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        var y = bounds.minY

        for row in rows {
            let rowHeight = row.map { subviews[$0].sizeThatFits(.unspecified).height }.max() ?? 0
            var x = bounds.minX

            for index in row {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(size)
                )
                x += size.width + spacing
            }

            y += rowHeight + spacing
        }
    }

    private func computeRows(proposal: ProposedViewSize, subviews: Subviews) -> [[Int]] {
        let maxWidth = proposal.width ?? .infinity
        var rows: [[Int]] = [[]]
        var currentRowWidth: CGFloat = 0

        for (index, subview) in subviews.enumerated() {
            let size = subview.sizeThatFits(.unspecified)

            if currentRowWidth + size.width > maxWidth && !rows[rows.count - 1].isEmpty {
                rows.append([])
                currentRowWidth = 0
            }

            rows[rows.count - 1].append(index)
            currentRowWidth += size.width + spacing
        }

        return rows
    }
}
