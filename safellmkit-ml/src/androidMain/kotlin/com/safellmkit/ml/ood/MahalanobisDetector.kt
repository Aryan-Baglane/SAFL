package com.safellmkit.ml.ood

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

internal object MahalanobisAssets {
    val mean: FloatArray
    val invCov: FloatArray

    init {
        val meanStream = MahalanobisAssets::class.java.getResourceAsStream("/assets/mean.npy")
            ?: error("mean.npy not found in assets")
        val invCovStream = MahalanobisAssets::class.java.getResourceAsStream("/assets/inv_cov.npy")
            ?: error("inv_cov.npy not found in assets")

        mean = parseNpy(meanStream.readBytes())
        invCov = parseNpy(invCovStream.readBytes())
    }

    private fun parseNpy(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        
        val magic = ByteArray(6)
        buffer.get(magic)
        if (magic[0] != 0x93.toByte() || String(magic, 1, 5) != "NUMPY") {
            error("Invalid npy magic")
        }
        val major = buffer.get().toInt() and 0xFF
        val minor = buffer.get().toInt() and 0xFF
        
        val headerLen = if (major == 1) {
            buffer.short.toInt() and 0xFFFF
        } else {
            buffer.int
        }
        
        val headerBytes = ByteArray(headerLen)
        buffer.get(headerBytes)
        val headerStr = String(headerBytes, StandardCharsets.UTF_8)
        
        val isDouble = headerStr.contains("'descr': '<f8'") || headerStr.contains("\"descr\": \"<f8\"") || headerStr.contains("'<f8'")
        
        val remainingBytes = bytes.size - buffer.position()
        if (isDouble) {
            val count = remainingBytes / 8
            val floats = FloatArray(count)
            for (i in 0 until count) {
                floats[i] = buffer.double.toFloat()
            }
            return floats
        } else {
            val count = remainingBytes / 4
            val floats = FloatArray(count)
            for (i in 0 until count) {
                floats[i] = buffer.float
            }
            return floats
        }
    }
}

actual class MahalanobisDetector actual constructor() {
    actual fun score(embedding: FloatArray): Float {
        val mean = MahalanobisAssets.mean
        val invCov = MahalanobisAssets.invCov

        if (embedding.size != mean.size) {
            return 0f
        }
        val diff = FloatArray(mean.size)
        for (i in mean.indices) {
            diff[i] = embedding[i] - mean[i]
        }
        val temp = FloatArray(mean.size)
        for (i in mean.indices) {
            var sum = 0.0
            val rowOffset = i * mean.size
            for (j in mean.indices) {
                sum += invCov[rowOffset + j] * diff[j]
            }
            temp[i] = sum.toFloat()
        }
        var dot = 0.0
        for (i in mean.indices) {
            dot += diff[i] * temp[i]
        }
        return kotlin.math.sqrt(dot).toFloat()
    }
}
