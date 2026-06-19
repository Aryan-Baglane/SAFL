package com.safellmkit.ml.model

interface RiskClassifier {
    fun predict(text: String, maxLength: Int = 256): RiskPrediction
    val supportsInference: Boolean
}
