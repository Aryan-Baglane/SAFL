package com.safellmkit.core.engine

actual fun loadCentroidsDiagnostics(): String {
    return try {
        val stream = GuardrailEngine::class.java.getResourceAsStream("/assets/class_centroids.npy")
            ?: return "Centroids not found in classpath assets"
        val bytes = stream.readBytes()
        if (bytes.size < 10) return "Centroids file too small"
        if (bytes[0] != 0x93.toByte() || bytes[1] != 'N'.toByte() || bytes[2] != 'U'.toByte() || bytes[3] != 'M'.toByte() || bytes[4] != 'P'.toByte() || bytes[5] != 'Y'.toByte()) {
            return "Invalid numpy header"
        }
        val headerLen = (bytes[8].toInt() and 0xFF) or ((bytes[9].toInt() and 0xFF) shl 8)
        val dataOffset = 10 + headerLen
        val dataBytes = bytes.size - dataOffset
        val numFloats = dataBytes / 4
        "Parsed class_centroids.npy: header len $headerLen, total bytes ${bytes.size}, data offset $dataOffset, floats count $numFloats"
    } catch (e: Throwable) {
        "Failed to load class centroids: ${e.message}"
    }
}

actual fun currentTimeMs(): Long = System.currentTimeMillis()
