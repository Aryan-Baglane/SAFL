package com.safellmkit.ml.temporal

@OptIn(kotlin.ExperimentalStdlibApi::class)
actual class TemporalRiskModel actual constructor(
    modelPath: String
) : AutoCloseable {

    actual constructor() : this("")

    actual val supportsInference: Boolean = false

    actual fun predict(sequence: List<FloatArray>): TemporalPrediction {
        return TemporalPrediction(
            crescendoProbability = 0f,
            pairProbability = 0f
        )
    }

    actual override fun close() {
        // No-op
    }
}
