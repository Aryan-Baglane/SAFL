package com.safellmkit.ml.temporal

@OptIn(kotlin.ExperimentalStdlibApi::class)
expect class TemporalRiskModel(modelPath: String) : AutoCloseable {
    constructor()
    fun predict(sequence: List<FloatArray>): TemporalPrediction
    val supportsInference: Boolean
    override fun close()
}
