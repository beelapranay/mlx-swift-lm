// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Offline architecture checks for the LFM3 port, built from a tiny synthetic config.
final class LFM3Tests: XCTestCase {

    private func configuration(
        layerTypes: String = #"["conv", "full_attention", "conv", "sliding_attention"]"#,
        ropeParameters: String = """
        {
            "full_attention": {
                "rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000}
        }
        """
    ) throws -> LFM3Configuration {
        let json = """
            {
                "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32,
                "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 128,
                "norm_eps": 1e-5, "conv_bias": false, "conv_L_cache": 3, "sliding_window": 4,
                "use_sandwich_norm": true, "tie_word_embeddings": true,
                "layer_types": \(layerTypes), "rope_parameters": \(ropeParameters)
            }
            """
        return try JSONDecoder().decode(LFM3Configuration.self, from: Data(json.utf8))
    }

    private func model(dtype: DType = .float32, seed: UInt64 = 7) throws -> LFM3Model {
        MLXRandom.seed(seed)
        let model = LFM3Model(try configuration())
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
        eval(model)
        return model
    }

    func testConfigurationBuildsPerLayerMixers() throws {
        let configuration = try configuration()
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
        let model = LFM3Model(configuration)
        let layers = model.model.layers

        XCTAssertEqual(layers.map(\.kind), [.conv, .full, .conv, .sliding])
        XCTAssertNotNil(layers[0].conv)
        XCTAssertNil(layers[0].attention)
        XCTAssertNotNil(layers[1].attention)
        XCTAssertNotNil(layers[1].postOperatorNorm)
        XCTAssertEqual(model.kvHeads, [0, 2, 0, 2])
        // RoPE frequencies are derived state, not checkpoint parameters.
        XCTAssertFalse(model.parameters().flattened().contains { $0.0.contains("rope") })
        XCTAssertEqual(model.toolCallFormat, .xmlFunction)
        XCTAssertEqual(model.reasoningConfig, QwenReasoningProtocol.tagged)
    }

    func testValidationRejectsInconsistentSchedules() throws {
        XCTAssertThrowsError(
            try configuration(layerTypes: #"["conv", "full_attention"]"#)
                .validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration(
                ropeParameters: #"{"full_attention": {"rope_theta": 10000.0}}"#
            ).validateModelConfiguration())
        XCTAssertThrowsError(
            try configuration(layerTypes: #"["conv", "global", "conv", "conv"]"#))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try model()
        let logits = model(MLXArray([Int32(0), 1, 2, 3, 4]).reshaped(1, 5), cache: nil)
        XCTAssertEqual(logits.shape, [1, 5, 64])
    }

    func testCachesMatchLayerTypes() throws {
        let caches = try model().newCache(parameters: nil)
        XCTAssertTrue(caches[0] is MambaCache)
        XCTAssertFalse(caches[1] is RotatingKVCache)
        XCTAssertTrue(caches[2] is MambaCache)
        XCTAssertTrue(caches[3] is RotatingKVCache)
        XCTAssertEqual(caches[3].maxSize, 4)
    }

    func testIncrementalDecodeMatchesFullForward() throws {
        let model = try model()
        // Eight positions wrap the four-token sliding window.
        let tokens = MLXArray([Int32(3), 9, 27, 5, 11, 2, 40, 17]).reshaped(1, 8)

        let full = model(tokens, cache: nil)

        let cache = try model.newCache(parameters: nil)
        let prefill = model(tokens[0..., ..<3], cache: cache)
        let steps = (3 ..< 8).map { model(tokens[0..., $0 ..< ($0 + 1)], cache: cache) }
        let incremental = concatenated([prefill] + steps, axis: 1)

        XCTAssertEqual(cache[1].offset, 8)
        XCTAssertEqual(cache[3].offset, 8)
        assertClose(
            incremental, full, rtol: 1e-4, atol: 1e-4, reducedAtol: 2e-2,
            "incremental decode diverged from the full forward")
    }

    func testPartialRoPELeavesTailUnrotated() throws {
        // head_dim 8 with factor 0.5 rotates pairs (0, 4) and (1, 5) only.
        let rope = LFM3RoPE(headDim: 8, base: 10_000, partialRotaryFactor: 0.5)
        MLXRandom.seed(3)
        let x = MLXRandom.normal([1, 1, 3, 8])
        let y = rope(x, offset: 5)

        let tail = MLXArray([Int32(2), 3, 6, 7])
        XCTAssertTrue(allClose(y.take(tail, axis: -1), x.take(tail, axis: -1)).item(Bool.self))
        XCTAssertFalse(allClose(y[.ellipsis, 0], x[.ellipsis, 0]).item(Bool.self))
    }

    func testSanitizeStripsPrefixAndTransposesConv() throws {
        let model = try model()
        let sanitized = model.sanitize(weights: [
            "language_model.model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 3]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([64, 32]),
            "model.layers.2.conv.conv.weight": MLXArray.zeros([32, 3, 1]),
        ])

        XCTAssertEqual(sanitized["model.layers.0.conv.conv.weight"]?.shape, [32, 3, 1])
        XCTAssertEqual(sanitized["model.layers.2.conv.conv.weight"]?.shape, [32, 3, 1])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNil(sanitized["language_model.model.embed_tokens.weight"])
    }

    func testRegisteredModelType() async throws {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "LiquidAI/LFM3-1.7B-MLX-4bit"))
        let json = Data(
            """
            {
                "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32,
                "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "norm_eps": 1e-5,
                "sliding_window": 4, "layer_types": ["conv", "sliding_attention"],
                "rope_parameters": {"sliding_attention": {"rope_theta": 10000}}
            }
            """.utf8)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: json, modelType: "lfm3")
        XCTAssertTrue(model is LFM3Model)
    }
}
