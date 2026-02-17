import SwiftUI

/// Displays the top-k next-token predictions with probability bars.
struct TopPredictionsView: View {
    let predictions: [TokenPrediction]

    private var maxProb: Float {
        predictions.first?.prob ?? 1
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Top Predictions", icon: "chart.bar.fill")

            VStack(spacing: 8) {
                ForEach(Array(predictions.prefix(5).enumerated()), id: \.offset) { index, prediction in
                    HStack(spacing: 12) {
                        // Rank
                        Text("#\(index + 1)")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.4))
                            .frame(width: 24, alignment: .trailing)

                        // Token text
                        Text(formatToken(prediction.token))
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white)
                            .frame(width: 100, alignment: .leading)

                        // Probability bar
                        GeometryReader { geo in
                            let fraction = maxProb > 0
                                ? CGFloat(prediction.prob / maxProb)
                                : 0

                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(.white.opacity(0.06))

                                RoundedRectangle(cornerRadius: 4)
                                    .fill(
                                        LinearGradient(
                                            colors: [
                                                .green.opacity(0.6),
                                                .green.opacity(0.3)
                                            ],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                                    .frame(width: max(fraction * geo.size.width, 4))
                            }
                        }
                        .frame(height: 20)

                        // Probability percentage
                        Text(String(format: "%.1f%%", prediction.prob * 100))
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.6))
                            .frame(width: 56, alignment: .trailing)
                    }
                }
            }
        }
        .cardBackground()
    }

    /// Formats special tokens for display (e.g. replaces whitespace markers).
    private func formatToken(_ token: String) -> String {
        token
            .replacingOccurrences(of: "\u2581", with: "\u2423") // ▁ → ␣
            .replacingOccurrences(of: "\n", with: "\\n")
    }
}
