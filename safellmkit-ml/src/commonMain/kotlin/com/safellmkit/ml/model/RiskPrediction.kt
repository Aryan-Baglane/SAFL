package com.safellmkit.ml.model

import kotlinx.serialization.Serializable

@Serializable
data class RiskPrediction(
    val probability: Float,
    val attackClass: AttackTaxonomy,
    val embedding: FloatArray,
    val logits: FloatArray,
    val uncertain: Boolean = false,
    val margin: Float = 0f,
    val entropy: Float = 0f
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as RiskPrediction

        if (probability != other.probability) return false
        if (attackClass != other.attackClass) return false
        if (!embedding.contentEquals(other.embedding)) return false
        if (!logits.contentEquals(other.logits)) return false
        if (uncertain != other.uncertain) return false
        if (margin != other.margin) return false
        if (entropy != other.entropy) return false

        return true
    }

    override fun hashCode(): Int {
        var result = probability.hashCode()
        result = 31 * result + attackClass.hashCode()
        result = 31 * result + embedding.contentHashCode()
        result = 31 * result + logits.contentHashCode()
        result = 31 * result + uncertain.hashCode()
        result = 31 * result + margin.hashCode()
        result = 31 * result + entropy.hashCode()
        return result
    }
}
