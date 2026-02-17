import Foundation

/// Result of tokenizing user input text via the C bridge.
struct TokenizationResult: Codable {
    let tokens: [String]
    let ids: [Int]
}

/// A single top-k predicted next token with its probability.
struct TokenPrediction: Codable, Identifiable {
    let id = UUID()
    let token: String
    let prob: Float

    enum CodingKeys: String, CodingKey {
        case token, prob
    }
}

/// Result of a debug forward pass through the model.
///
/// Heatmaps are nested arrays: outer = token positions, inner = activation dimensions
/// (downsampled to `heatmapCols` columns by the C bridge).
struct ForwardDebugData: Codable {
    let nLayers: Int
    let dModel: Int
    let seqLen: Int
    let embedNorms: [Float]
    let blockNorms: [[Float]]
    let embedHeatmap: [[Float]]
    let blockHeatmaps: [[[Float]]]
    let finalNormHeatmap: [[Float]]
    let heatmapCols: Int
    let topLogits: [TokenPrediction]
    let topLogitsPerPosition: [[TokenPrediction]]

    enum CodingKeys: String, CodingKey {
        case nLayers = "n_layers"
        case dModel = "d_model"
        case seqLen = "seq_len"
        case embedNorms = "embed_norms"
        case blockNorms = "block_norms"
        case embedHeatmap = "embed_heatmap"
        case blockHeatmaps = "block_heatmaps"
        case finalNormHeatmap = "final_norm_heatmap"
        case heatmapCols = "heatmap_cols"
        case topLogits = "top_logits"
        case topLogitsPerPosition = "top_logits_per_position"
    }
}

/// Model architecture configuration returned from the C bridge.
struct ModelArchConfig: Codable {
    let nLayers: Int
    let dModel: Int
    let dReduce: Int
    let dFf: Int
    let vocabSize: Int
    let maxSeqLen: Int

    enum CodingKeys: String, CodingKey {
        case nLayers = "n_layers"
        case dModel = "d_model"
        case dReduce = "d_reduce"
        case dFf = "d_ff"
        case vocabSize = "vocab_size"
        case maxSeqLen = "max_seq_len"
    }
}

/// Encapsulates all data needed for the interpretability overlay,
/// differentiated by whether the tapped message was from the user or assistant.
enum InterpretabilityPayload {
    case userMessage(tokenization: TokenizationResult, debugData: ForwardDebugData)
    case assistantMessage(
        tokenization: TokenizationResult,
        debugData: ForwardDebugData,
        archConfig: ModelArchConfig
    )
}
