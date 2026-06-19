package com.safellmkit.ml.ood

expect class MahalanobisDetector() {
    fun score(embedding: FloatArray): Float
}
