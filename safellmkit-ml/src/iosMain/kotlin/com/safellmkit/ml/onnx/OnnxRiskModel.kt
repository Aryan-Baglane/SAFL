package com.safellmkit.ml.onnx

import com.safellmkit.ml.model.AttackTaxonomy
import com.safellmkit.ml.model.RiskClassifier
import com.safellmkit.ml.model.RiskPrediction
import com.safellmkit.ml.tokenizer.TextTokenizer
import com.safellmkit.ml.tokenizer.TokenizedInput

@OptIn(kotlin.ExperimentalStdlibApi::class)
actual class OnnxRiskModel actual constructor(
    modelPath: String,
    private val tokenizer: TextTokenizer
) : RiskClassifier, AutoCloseable {

    actual constructor() : this("", object : TextTokenizer {
        override fun encode(text: String, maxLength: Int): TokenizedInput {
            return TokenizedInput(LongArray(0), LongArray(0))
        }
    })

    override actual val supportsInference: Boolean = false

    override actual fun predict(text: String, maxLength: Int): RiskPrediction {
        return RiskPrediction(
            probability = 0f,
            attackClass = AttackTaxonomy.SAFE,
            embedding = FloatArray(384),
            logits = FloatArray(9)
        )
    }

    actual fun predictAttackProbability(text: String, maxLength: Int): Float {
        return 0f
    }

    override actual fun close() {
        // No-op
    }
}