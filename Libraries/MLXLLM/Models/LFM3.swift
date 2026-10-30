//
//  LFM3.swift
//  mlx-swift-lm
//

// Port of mlx_lm/models/lfm3.py from https://github.com/ml-explore/mlx-lm/pull/2016

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct LFM3Configuration: Codable, Sendable {
    public enum LayerKind: String, Codable, Sendable {
        case conv
        case full = "full_attention"
        case sliding = "sliding_attention"
    }

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var normEps: Float
    var convBias: Bool
    var convLCache: Int
    var layerTypes: [LayerKind]
    var slidingWindow: Int
    var ropeParameters: [String: [String: StringOrNumber]]
    var useSandwichNorm: Bool
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case ropeParameters = "rope_parameters"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3"
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        normEps = try container.decode(Float.self, forKey: .normEps)
        convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        layerTypes = try container.decode([LayerKind].self, forKey: .layerTypes)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        ropeParameters = try container.decode(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        useSandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    /// RoPE base and partial rotary factor of one attention kind.
    func rope(for kind: LayerKind) -> (theta: Float, partialRotaryFactor: Float) {
        let parameters = ropeParameters[kind.rawValue]
        return (
            parameters?["rope_theta"]?.asFloat() ?? 10_000,
            parameters?["partial_rotary_factor"]?.asFloat() ?? 1.0
        )
    }
}

extension LFM3Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard layerTypes.count == hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "layer_types needs \(hiddenLayers) entries, got \(layerTypes.count)")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "num_attention_heads (\(attentionHeads)) must be a multiple of "
                    + "num_key_value_heads (\(kvHeads))")
        }
        for kind in Set(layerTypes) where kind != .conv {
            guard let parameters = ropeParameters[kind.rawValue] else {
                throw ModelFactoryError.invalidConfiguration(
                    "rope_parameters has no entry for \(kind.rawValue)")
            }
            try validateRoPEConfiguration(
                parameters, context: "LFM3Configuration.rope_parameters.\(kind.rawValue)",
                supportedTypes: ["default"])
        }
    }
}

// MARK: - RoPE

/// RoPE over the full head dimension that leaves the tail unrotated when
/// `partialRotaryFactor < 1`. The rotated pairs keep the full-dimension frequencies.
final class LFM3RoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    // The leading underscore keeps this out of the checkpoint parameters.
    private let _freqs: MLXArray

    init(headDim: Int, base: Float, partialRotaryFactor: Float) {
        self.dimensions = headDim
        let pairs = headDim / 2
        let rotatedPairs = Int(partialRotaryFactor * Float(headDim)) / 2
        let exponents =
            MLXArray(stride(from: 0, to: 2 * rotatedPairs, by: 2)).asType(.float32)
            / Float(headDim)
        var freqs = pow(MLXArray(base), exponents)
        if rotatedPairs < pairs {
            // An effectively infinite wavelength rotates by zero.
            freqs = concatenated([
                freqs, MLXArray.full([pairs - rotatedPairs], values: MLXArray(Float(1e30))),
            ])
        }
        self._freqs = freqs
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: nil, scale: 1.0,
            offset: offset, freqs: _freqs)
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: nil, scale: 1.0,
            offset: offset, freqs: _freqs)
    }
}

// MARK: - Attention

class LFM3Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: LFM3RoPE

    init(_ args: LFM3Configuration, kind: LFM3Configuration.LayerKind) {
        let dim = args.hiddenSize
        let headDim = args.headDim
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(dim, attentionHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(attentionHeads * headDim, dim, bias: false)
        _gateProj.wrappedValue = Linear(dim, attentionHeads, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

        let rope = args.rope(for: kind)
        self.rope = LFM3RoPE(
            headDim: headDim, base: rope.theta, partialRotaryFactor: rope.partialRotaryFactor)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, attentionHeads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)

        // Sigmoid output gate, one scalar per head
        let gate = sigmoid(gateProj(x))
        return oProj((output * gate.expandedDimensions(axis: -1)).reshaped(B, L, -1))
    }
}

// MARK: - Short convolution

class LFM3ShortConv: Module {
    let hiddenSize: Int
    let lCache: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        self.hiddenSize = args.hiddenSize
        self.lCache = args.convLCache

        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: lCache,
            groups: hiddenSize, bias: args.convBias)
        _inProj.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let parts = inProj(x).split(parts: 3, axis: -1)
        let (b, c, h) = (parts[0], parts[1], parts[2])
        var bx = b * h

        let state =
            cache?[0]
            ?? MLXArray.zeros([bx.dim(0), lCache - 1, hiddenSize], dtype: bx.dtype)
        bx = concatenated([state, bx], axis: -2)
        if let cache {
            cache[0] = contiguous(bx[0..., (bx.dim(1) - (lCache - 1))..., 0...])
            cache.advance(x.dim(1))
        }

        return outProj(c * conv(bx))
    }
}

// MARK: - Decoder

class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(_ args: LFM3Configuration) {
        _w1.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        _w2.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        _w3.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

class LFM3DecoderLayer: Module {
    let kind: LFM3Configuration.LayerKind

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ args: LFM3Configuration, layerIdx: Int) {
        self.kind = args.layerTypes[layerIdx]

        if kind == .conv {
            _conv.wrappedValue = LFM3ShortConv(args)
        } else {
            _attention.wrappedValue = LFM3Attention(args, kind: kind)
        }
        _feedForward.wrappedValue = LFM3MLP(args)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if args.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r: MLXArray
        if let attention {
            r = attention(operatorNorm(x), mask: mask, cache: cache)
        } else {
            r = conv!(operatorNorm(x), cache: cache as? MambaCache)
        }
        if let postOperatorNorm {
            r = postOperatorNorm(r)
        }
        let h = x + r

        r = feedForward(ffnNorm(h))
        if let postFFNNorm {
            r = postFFNNorm(r)
        }
        return h + r
    }
}

public class LFM3ModelInner: Module {
    let slidingWindow: Int
    let fullIdx: Int?
    let slidingIdx: Int?

    let layers: [LFM3DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ args: LFM3Configuration) {
        precondition(args.vocabularySize > 0)

        self.slidingWindow = args.slidingWindow
        self.fullIdx = args.layerTypes.firstIndex(of: .full)
        self.slidingIdx = args.layerTypes.firstIndex(of: .sliding)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { LFM3DecoderLayer(args, layerIdx: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: fullIdx.flatMap { cache?[$0] })
        let slidingMask = createAttentionMask(
            h: h, cache: slidingIdx.flatMap { cache?[$0] }, windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                switch layer.kind {
                case .full: fullMask
                case .sliding: slidingMask
                case .conv: .none
                }
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return embeddingNorm(h)
    }
}

public class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM3ModelInner
    let configuration: LFM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0 == .conv ? 0 : args.kvHeads }
        self.model = LFM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead?(out) ?? model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            // Converted checkpoints keep the VLM-style prefix
            let name =
                key.hasPrefix("language_model.")
                ? String(key.dropFirst("language_model.".count)) : key
            var param = value
            if name.contains("conv.weight"), param.dim(-1) > param.dim(1) {
                param = param.transposed(0, 2, 1)
            }
            sanitized[name] = param
        }
        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.layerTypes.map { kind in
            switch kind {
            case .conv:
                MambaCache()
            case .full, .sliding:
                try makeHybridAttentionKVCache(
                    parameters: parameters,
                    slidingWindow: configuration.slidingWindow,
                    usesSlidingWindow: kind == .sliding)
            }
        }
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension LFM3Model {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
