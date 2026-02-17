import SwiftUI

/// Classic neural network diagram with circle neurons and connection lines.
///
/// Layers are arranged left-to-right: Input → Embed → Block 0..N-1 → Final LN → Output.
/// Neurons and connections light up with brightness proportional to actual per-layer,
/// per-neuron activation norms from the forward pass at the current token position.
///
/// The animation runs one pass per token, showing how the model processes each token
/// in sequence. Generated tokens appear in the output box as each pass completes,
/// and per-step predictions are shown below.
///
/// Playback controls: play/pause, reset, step forward/backward, and speed slider.
struct NeuralNetworkDiagramView: View {
    let debugData: ForwardDebugData
    let archConfig: ModelArchConfig
    let tokens: [String]

    /// Pre-computed once in init — avoids O(n) iteration over all norms on every frame.
    private let globalMaxNorm: Float

    @State private var animationProgress: CGFloat = 0
    @State private var currentTokenStep: Int = 0
    @State private var isPlaying = false
    @State private var speed: Double = 1.0
    @State private var hasAppeared = false
    @State private var generatedTokens: [String] = []

    /// 60 fps timer driving the animation when playing.
    private let frameTimer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()

    init(debugData: ForwardDebugData, archConfig: ModelArchConfig, tokens: [String]) {
        self.debugData = debugData
        self.archConfig = archConfig
        self.tokens = tokens

        var maxVal: Float = 0
        for n in debugData.embedNorms { maxVal = max(maxVal, abs(n)) }
        for layer in debugData.blockNorms {
            for n in layer { maxVal = max(maxVal, abs(n)) }
        }
        for row in debugData.finalNormHeatmap {
            let n = sqrt(row.reduce(0) { $0 + $1 * $1 })
            maxVal = max(maxVal, n)
        }
        self.globalMaxNorm = max(maxVal, 1e-6)
    }

    // MARK: - Visual Constants

    private let neuronRadius: CGFloat = 9
    private let inputNeuronCount = 5
    private let hiddenNeuronCount = 7
    private let outputNeuronCount = 3
    private let diagramHeight: CGFloat = 350
    private let horizontalPadding: CGFloat = 40
    private let topPadding: CGFloat = 26
    private let bottomPadding: CGFloat = 14

    // MARK: - Computed Properties

    private var totalTokenSteps: Int {
        max(min(tokens.count, debugData.seqLen), 1)
    }

    /// Whether the entire animation has finished (all token steps complete).
    private var animationComplete: Bool {
        currentTokenStep >= totalTokenSteps - 1 && animationProgress >= 1.0
    }


    // MARK: - Token Display

    /// Reconstruct displayable text from raw WordPiece tokens.
    /// Non-## tokens get a space before them (except the first); ## tokens are joined directly.
    private var displayText: String {
        var result = ""
        for (i, token) in generatedTokens.enumerated() {
            if token.hasPrefix("##") {
                // WordPiece continuation: append without space, strip prefix
                result += String(token.dropFirst(2))
            } else if token.hasPrefix("\u{2581}") {
                // SentencePiece space marker: replace with actual space
                if i == 0 {
                    result += String(token.dropFirst())
                } else {
                    result += " " + String(token.dropFirst())
                }
            } else if i > 0 {
                result += " " + token
            } else {
                result += token
            }
        }
        return result
    }

    /// Format a single token for display in predictions/indicators (shows special chars visually).
    private func formatTokenForDisplay(_ token: String) -> String {
        token
            .replacingOccurrences(of: "\u{2581}", with: "\u{2423}") // ▁ → ␣
            .replacingOccurrences(of: "\n", with: "\\n")
    }

    // MARK: - Layer Configuration

    private var layers: [LayerInfo] {
        let t = min(currentTokenStep, max(debugData.seqLen - 1, 0))
        var result = [LayerInfo]()

        // Input layer — fixed brightness, no real activation data
        result.append(LayerInfo(
            label: "Input",
            neuronCount: inputNeuronCount,
            normFactor: 0.7,
            neuronActivations: nil
        ))

        // Embed layer — per-position norm + per-neuron heatmap
        let embedNorm = t < debugData.embedNorms.count ? debugData.embedNorms[t] : 0
        let embedAct = t < debugData.embedHeatmap.count
            ? normalizedNeuronActivations(debugData.embedHeatmap[t], count: hiddenNeuronCount)
            : nil
        result.append(LayerInfo(
            label: "Embed",
            neuronCount: hiddenNeuronCount,
            normFactor: normToFactor(embedNorm),
            neuronActivations: embedAct
        ))

        // Block layers — per-position per-layer norms + heatmaps
        for i in 0..<archConfig.nLayers {
            let norm: Float = (i < debugData.blockNorms.count && t < debugData.blockNorms[i].count)
                ? debugData.blockNorms[i][t] : 0
            let act: [Float]? = (i < debugData.blockHeatmaps.count && t < debugData.blockHeatmaps[i].count)
                ? normalizedNeuronActivations(debugData.blockHeatmaps[i][t], count: hiddenNeuronCount)
                : nil
            result.append(LayerInfo(
                label: "Block \(i)",
                neuronCount: hiddenNeuronCount,
                normFactor: normToFactor(norm),
                neuronActivations: act
            ))
        }

        // Final LN
        if t < debugData.finalNormHeatmap.count {
            let row = debugData.finalNormHeatmap[t]
            let norm = sqrt(row.reduce(0) { $0 + $1 * $1 })
            result.append(LayerInfo(
                label: "LN",
                neuronCount: hiddenNeuronCount,
                normFactor: normToFactor(norm),
                neuronActivations: normalizedNeuronActivations(row, count: hiddenNeuronCount)
            ))
        } else {
            result.append(LayerInfo(
                label: "LN",
                neuronCount: hiddenNeuronCount,
                normFactor: 0,
                neuronActivations: nil
            ))
        }

        // Output layer — show real prediction data only on the last step
        result.append(LayerInfo(
            label: "Output",
            neuronCount: outputNeuronCount,
            normFactor: 0.7,
            neuronActivations: outputNeuronActivations()
        ))

        return result
    }

    /// Convert raw norm to a display factor with high contrast.
    /// Non-zero norms map to [0.05, 1.0]; zero norms map to 0.
    private func normToFactor(_ norm: Float) -> CGFloat {
        guard abs(norm) > 1e-8 else { return 0 }
        let raw = abs(norm) / globalMaxNorm
        return CGFloat(0.05 + 0.95 * pow(raw, 0.6))
    }

    /// Downsample a heatmap row to `count` display neurons using absolute values normalized to [0, 1].
    private func normalizedNeuronActivations(_ row: [Float], count: Int) -> [Float] {
        guard !row.isEmpty else { return Array(repeating: 0, count: count) }
        let binSize = max(1, row.count / count)
        var values = (0..<count).map { j -> Float in
            let start = j * binSize
            let end = min(start + binSize, row.count)
            guard end > start else { return 0 }
            return row[start..<end].reduce(0) { $0 + abs($1) } / Float(end - start)
        }
        let maxVal = values.max() ?? 0
        if maxVal > 1e-6 {
            values = values.map { $0 / maxVal }
        }
        return values
    }

    /// Map top prediction probabilities to output display neurons (last step only).
    private func outputNeuronActivations() -> [Float]? {
        guard currentTokenStep >= totalTokenSteps - 1 else { return nil }
        let preds = debugData.topLogits
        guard !preds.isEmpty else { return nil }
        let maxProb = preds.map(\.prob).max() ?? 1
        return (0..<outputNeuronCount).map { j in
            j < preds.count ? preds[j].prob / max(maxProb, 1e-6) : 0
        }
    }

    private var layerThresholds: [CGFloat] {
        let count = layers.count
        guard count > 1 else { return [0] }
        return (0..<count).map { CGFloat($0) / CGFloat(count - 1) }
    }

    // MARK: - Body

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeader("Architecture", icon: "cpu")

            controlsBar

            // Token step indicator
            HStack(spacing: 6) {
                Text("Pass \(currentTokenStep + 1)/\(totalTokenSteps)")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))

                if currentTokenStep < tokens.count {
                    Text(formatTokenForDisplay(tokens[currentTokenStep]))
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.green.opacity(0.8))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(.green.opacity(0.1))
                        )
                }
            }

            Canvas { context, size in
                drawNetwork(in: &context, size: size)
            }
            .frame(height: diagramHeight)

            generatedOutputBox

            predictionsBox
        }
        .cardBackground()
        .onReceive(frameTimer) { _ in
            guard isPlaying else { return }
            let perStepDuration = max(0.6, 2.0 - Double(totalTokenSteps) * 0.04)
            let increment = CGFloat(1.0 / 60.0) * CGFloat(speed) / CGFloat(perStepDuration)
            animationProgress = min(1.0, animationProgress + increment)
            if animationProgress >= 1.0 {
                // Current step complete — add token to output
                if currentTokenStep < tokens.count {
                    generatedTokens.append(tokens[currentTokenStep])
                }
                // Advance to next token step or stop
                if currentTokenStep < totalTokenSteps - 1 {
                    currentTokenStep += 1
                    animationProgress = 0
                } else {
                    isPlaying = false
                }
            }
        }
        .onAppear {
            guard !hasAppeared else { return }
            hasAppeared = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                isPlaying = true
            }
        }
    }

    // MARK: - Generated Output Box

    private var generatedOutputBox: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "text.cursor")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
                Text("Generated Output")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
            }

            HStack(spacing: 0) {
                if generatedTokens.isEmpty {
                    Text("...")
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.2))
                } else {
                    Text(displayText)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.green)
                        .lineLimit(nil)
                        .fixedSize(horizontal: false, vertical: true)
                }

                // Cursor shown while animation is ongoing
                if !animationComplete {
                    Rectangle()
                        .fill(.green.opacity(0.7))
                        .frame(width: 2, height: 16)
                }

                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(.black.opacity(0.3))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(.white.opacity(0.08), lineWidth: 1)
                    )
            )
        }
    }

    // MARK: - Predictions

    /// Predictions that explain the most recently generated token.
    /// tokens[t] was predicted by position t-1's logits, so we use generatedTokens.count - 2.
    /// Updates only when a new token appears in the output box.
    private var currentPredictions: [TokenPrediction] {
        let count = generatedTokens.count
        guard count >= 2 else { return [] }
        let pos = count - 2
        guard pos < debugData.topLogitsPerPosition.count else { return [] }
        return debugData.topLogitsPerPosition[pos]
    }

    /// The most recently generated token — highlighted in the predictions list
    /// as the one that was "chosen" from the candidates.
    private var nextChosenToken: String? {
        guard generatedTokens.count >= 2 else { return nil }
        return generatedTokens.last
    }

    @ViewBuilder
    private var predictionsBox: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "chart.bar.fill")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
                Text("Token Probabilities")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))

                Spacer()

                if generatedTokens.count >= 2 {
                    Text("pos \(generatedTokens.count - 2)")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.3))
                }
            }

            let predictions = currentPredictions
            if predictions.isEmpty {
                Text("...")
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.2))
            } else {
                predictionRows(predictions)
            }
        }
    }

    private func predictionRows(_ predictions: [TokenPrediction]) -> some View {
        let maxProb = predictions.first?.prob ?? 1
        let chosen = nextChosenToken

        return VStack(spacing: 6) {
            ForEach(Array(predictions.prefix(5).enumerated()), id: \.offset) { index, prediction in
                let isChosen = chosen != nil && prediction.token == chosen

                HStack(spacing: 10) {
                    // Rank or checkmark for the chosen prediction
                    if isChosen {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(.green)
                            .frame(width: 20, alignment: .trailing)
                    } else {
                        Text("#\(index + 1)")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.4))
                            .frame(width: 20, alignment: .trailing)
                    }

                    // Token text
                    Text(formatTokenForDisplay(prediction.token))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(isChosen ? .green : .white)
                        .frame(width: 80, alignment: .leading)

                    // Probability bar — full width, no animation scaling
                    GeometryReader { geo in
                        let fraction = maxProb > 0
                            ? CGFloat(prediction.prob / maxProb) : 0
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(.white.opacity(0.06))
                            RoundedRectangle(cornerRadius: 3)
                                .fill(
                                    LinearGradient(
                                        colors: isChosen
                                            ? [.green.opacity(0.8), .green.opacity(0.5)]
                                            : [.green.opacity(0.6), .green.opacity(0.3)],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: max(fraction * geo.size.width, 3))
                        }
                    }
                    .frame(height: 16)

                    // Probability percentage
                    Text(String(format: "%.1f%%", prediction.prob * 100))
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(isChosen ? .green.opacity(0.8) : .white.opacity(0.5))
                        .frame(width: 48, alignment: .trailing)
                }
                .padding(.horizontal, isChosen ? 8 : 0)
                .padding(.vertical, isChosen ? 3 : 0)
                .background(
                    Group {
                        if isChosen {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(.green.opacity(0.08))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .strokeBorder(.green.opacity(0.2), lineWidth: 1)
                                )
                        }
                    }
                )
            }
        }
        .animation(.easeOut(duration: 0.25), value: generatedTokens.count)
    }

    // MARK: - Playback Controls

    private var controlsBar: some View {
        HStack(spacing: 12) {
            // Grouped playback buttons
            HStack(spacing: 2) {
                controlButton("backward.end.fill", action: stepBackward)
                controlButton(
                    isPlaying ? "pause.fill" : "play.fill",
                    tint: isPlaying ? .green : nil,
                    action: togglePlayback
                )
                controlButton("forward.end.fill", action: stepForward)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(.white.opacity(0.06))
            )

            // Reset
            controlButton("arrow.counterclockwise", action: resetAnimation)

            Spacer()

            // Speed slider
            HStack(spacing: 6) {
                let label = speed == floor(speed)
                    ? String(format: "%.0fx", speed)
                    : String(format: "%.1fx", speed)
                Text(label)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
                    .frame(width: 28, alignment: .trailing)

                Slider(value: $speed, in: 0.5...5.0, step: 0.5)
                    .frame(width: 100)
                    .tint(.green.opacity(0.6))
            }
        }
    }

    private func controlButton(
        _ icon: String,
        tint: Color? = nil,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(tint ?? .white.opacity(0.7))
                .frame(width: 24, height: 24)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { inside in
            if inside { NSCursor.pointingHand.push() } else { NSCursor.pop() }
        }
    }

    // MARK: - Playback Actions

    private func togglePlayback() {
        if isPlaying {
            isPlaying = false
        } else {
            // If at the very end, reset everything before replaying
            if animationComplete {
                currentTokenStep = 0
                animationProgress = 0
                generatedTokens.removeAll()
            }
            isPlaying = true
        }
    }

    private func resetAnimation() {
        isPlaying = false
        currentTokenStep = 0
        generatedTokens.removeAll()
        withAnimation(.easeOut(duration: 0.2)) {
            animationProgress = 0
        }
    }

    private func stepForward() {
        isPlaying = false
        let thresholds = layerThresholds

        // Try to advance within current pass
        for t in thresholds where t > animationProgress + 0.001 {
            withAnimation(.easeInOut(duration: 0.25)) {
                animationProgress = t
            }
            return
        }

        // End of current pass — advance to next token step
        if currentTokenStep < tokens.count {
            generatedTokens.append(tokens[currentTokenStep])
        }
        if currentTokenStep < totalTokenSteps - 1 {
            currentTokenStep += 1
            withAnimation(.easeInOut(duration: 0.25)) {
                animationProgress = 0
            }
        } else {
            withAnimation(.easeInOut(duration: 0.25)) {
                animationProgress = 1.0
            }
        }
    }

    private func stepBackward() {
        isPlaying = false
        let thresholds = layerThresholds

        // Try to go back within current pass
        for t in thresholds.reversed() where t < animationProgress - 0.001 {
            withAnimation(.easeInOut(duration: 0.25)) {
                animationProgress = t
            }
            return
        }

        // At start of pass — go back to previous token step
        if currentTokenStep > 0 {
            if !generatedTokens.isEmpty {
                generatedTokens.removeLast()
            }
            currentTokenStep -= 1
            withAnimation(.easeInOut(duration: 0.25)) {
                animationProgress = 1.0
            }
        } else {
            withAnimation(.easeInOut(duration: 0.25)) {
                animationProgress = 0
            }
        }
    }

    // MARK: - Canvas Drawing

    private func drawNetwork(in context: inout GraphicsContext, size: CGSize) {
        let infos = layers
        let layerCount = infos.count
        guard layerCount > 1 else { return }

        let drawWidth = size.width - 2 * horizontalPadding
        let neuronTop = topPadding
        let neuronHeight = size.height - topPadding - bottomPadding

        func layerX(_ i: Int) -> CGFloat {
            horizontalPadding + drawWidth * CGFloat(i) / CGFloat(layerCount - 1)
        }

        // Pre-compute all layer positions once (avoids 2-3x redundant calculation per layer)
        let allPositions: [[CGPoint]] = (0..<layerCount).map { i in
            let x = layerX(i)
            let n = infos[i].neuronCount
            let spacing = neuronHeight / CGFloat(n + 1)
            return (0..<n).map { j in
                CGPoint(x: x, y: neuronTop + spacing * CGFloat(j + 1))
            }
        }

        // --- Connections ---
        for i in 0..<(layerCount - 1) {
            let fromPts = allPositions[i]
            let toPts = allPositions[i + 1]
            let glow = connectionGlow(fromLayer: i, totalLayers: layerCount)
            let fromInfo = infos[i]
            let toInfo = infos[i + 1]

            for (fi, f) in fromPts.enumerated() {
                for (ti, t) in toPts.enumerated() {
                    var path = Path()
                    path.move(to: f)
                    path.addLine(to: t)

                    if glow > 0.01 {
                        // Connection brightness based on connected neuron activations
                        let fromAct: Float = (fromInfo.neuronActivations != nil && fi < fromInfo.neuronActivations!.count)
                            ? fromInfo.neuronActivations![fi] : 0.5
                        let toAct: Float = (toInfo.neuronActivations != nil && ti < toInfo.neuronActivations!.count)
                            ? toInfo.neuronActivations![ti] : 0.5
                        let avgNormFactor = (fromInfo.normFactor + toInfo.normFactor) / 2.0
                        let strength = CGFloat((fromAct + toAct) / 2.0) * avgNormFactor
                        let opacity = (0.01 + 0.45 * strength) * glow
                        context.stroke(
                            path,
                            with: .color(.green.opacity(opacity)),
                            lineWidth: 0.3 + 1.2 * strength * glow
                        )
                    } else {
                        context.stroke(
                            path,
                            with: .color(.white.opacity(0.02)),
                            lineWidth: 0.4
                        )
                    }
                }
            }
        }

        // --- Neurons ---
        for i in 0..<layerCount {
            let pts = allPositions[i]
            let info = infos[i]
            let timingGlow = neuronGlow(layer: i, totalLayers: layerCount)

            for (j, pos) in pts.enumerated() {
                // Per-neuron activation from heatmap data
                let perNeuronFactor: CGFloat
                if let activations = info.neuronActivations, j < activations.count {
                    perNeuronFactor = CGFloat(activations[j])
                } else {
                    // Fallback: slight position-based variation
                    let mid = CGFloat(pts.count - 1) / 2.0
                    let dist = abs(CGFloat(j) - mid) / max(mid, 1)
                    perNeuronFactor = 0.75 + 0.25 * (1 - dist)
                }

                let intensity = info.normFactor * timingGlow * perNeuronFactor

                // Glow halo
                if intensity > 0.1 {
                    let gs = neuronRadius * 2.8
                    context.fill(
                        Circle().path(in: CGRect(x: pos.x - gs, y: pos.y - gs, width: gs * 2, height: gs * 2)),
                        with: .color(.green.opacity(0.22 * intensity))
                    )
                }

                // Neuron body — dark base for low activations, vivid green for high
                let rect = CGRect(
                    x: pos.x - neuronRadius,
                    y: pos.y - neuronRadius,
                    width: neuronRadius * 2,
                    height: neuronRadius * 2
                )
                let fill = Color(
                    red: 0.04 + 0.12 * intensity,
                    green: 0.04 + 0.82 * intensity,
                    blue: 0.04 + 0.14 * intensity
                )
                context.fill(Circle().path(in: rect), with: .color(fill))

                // Border — barely visible when dim, bright when active
                context.stroke(
                    Circle().path(in: rect),
                    with: .color(.white.opacity(0.06 + 0.74 * intensity)),
                    lineWidth: 0.8 + 0.8 * intensity
                )
            }
        }

        // --- Output arrows ---
        let outputIdx = layerCount - 1
        let outputActivation = neuronGlow(layer: outputIdx, totalLayers: layerCount)
        for pos in allPositions[outputIdx] {
            let start = CGPoint(x: pos.x + neuronRadius + 3, y: pos.y)
            let end = CGPoint(x: pos.x + neuronRadius + 16, y: pos.y)

            var arrow = Path()
            arrow.move(to: start)
            arrow.addLine(to: end)
            arrow.move(to: CGPoint(x: end.x - 4, y: end.y - 3))
            arrow.addLine(to: end)
            arrow.addLine(to: CGPoint(x: end.x - 4, y: end.y + 3))

            context.stroke(
                arrow,
                with: .color(.white.opacity(0.25 + 0.5 * outputActivation)),
                lineWidth: 1.2
            )
        }

        // --- Layer labels ---
        for (i, info) in infos.enumerated() {
            let x = layerX(i)
            let resolved = context.resolve(
                Text(info.label)
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundColor(.white.opacity(0.5))
            )
            context.draw(resolved, at: CGPoint(x: x, y: 8), anchor: .top)
        }
    }

    // MARK: - Animation Helpers

    private func neuronGlow(layer: Int, totalLayers: Int) -> CGFloat {
        let threshold = CGFloat(layer) / CGFloat(totalLayers - 1)
        let spread: CGFloat = 0.12
        if animationProgress >= threshold + spread { return 1.0 }
        if animationProgress > threshold - spread {
            return (animationProgress - threshold + spread) / (2 * spread)
        }
        return 0
    }

    private func connectionGlow(fromLayer: Int, totalLayers: Int) -> CGFloat {
        let mid = (CGFloat(fromLayer) + 0.5) / CGFloat(totalLayers - 1)
        let spread: CGFloat = 0.10
        if animationProgress >= mid + spread { return 1.0 }
        if animationProgress > mid - spread {
            return (animationProgress - mid + spread) / (2 * spread)
        }
        return 0
    }
}

// MARK: - Layer Info

private struct LayerInfo {
    let label: String
    let neuronCount: Int
    let normFactor: CGFloat
    let neuronActivations: [Float]?
}
