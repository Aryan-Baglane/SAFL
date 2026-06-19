package com.safellmkit.ml.temporal

import kotlinx.serialization.Serializable

@Serializable
data class TemporalPrediction(
    val crescendoProbability: Float,
    val pairProbability: Float
)
