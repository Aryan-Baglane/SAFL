package com.safellmkit.ml.onnx

import com.safellmkit.ml.model.RiskClassifier
import com.safellmkit.ml.model.RiskPrediction
import com.safellmkit.ml.tokenizer.TextTokenizer

@OptIn(kotlin.ExperimentalStdlibApi::class)
expect class OnnxRiskModel(
    modelPath: String,
    tokenizer: TextTokenizer
) : RiskClassifier, AutoCloseable {
    constructor()
    override fun predict(text: String, maxLength: Int): RiskPrediction
    fun predictAttackProbability(text: String, maxLength: Int = 256): Float
    override val supportsInference: Boolean
    override fun close()
}
