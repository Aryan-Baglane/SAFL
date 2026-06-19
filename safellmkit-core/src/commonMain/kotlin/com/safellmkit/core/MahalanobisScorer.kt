package com.safellmkit.core.engine

import kotlin.math.sqrt

class MahalanobisScorer(
    private val mean: FloatArray,
    private val inverseCovariance: Array<FloatArray>
) {
    fun score(embedding: FloatArray): Float {
        val n = minOf(mean.size, embedding.size)
        val diff = FloatArray(n)
        for (i in 0 until n) {
            diff[i] = embedding[i] - mean[i]
        }

        val tmp = FloatArray(n)
        for (i in 0 until n) {
            var sum = 0f
            for (j in 0 until n) {
                sum += inverseCovariance[i][j] * diff[j]
            }
            tmp[i] = sum
        }

        var quad = 0f
        for (i in 0 until n) {
            quad += diff[i] * tmp[i]
        }

        return sqrt(quad.coerceAtLeast(0f))
    }
}