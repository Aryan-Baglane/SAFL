package com.safellmkit.ml.temporal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

@OptIn(kotlin.ExperimentalStdlibApi::class)
actual class TemporalRiskModel actual constructor(
    modelPath: String
) : AutoCloseable {

    actual constructor() : this(
        modelPath = kotlin.run {
            val localFile = File("outputs/temporal_model.onnx")
            if (localFile.exists()) {
                localFile.absolutePath
            } else {
                val stream = TemporalRiskModel::class.java.getResourceAsStream("/assets/temporal_model.onnx")
                    ?: error("temporal_model.onnx not found at outputs/temporal_model.onnx or in classpath assets")
                val tempFile = File.createTempFile("temporal_model", ".onnx")
                tempFile.deleteOnExit()
                tempFile.outputStream().use { out -> stream.copyTo(out) }
                tempFile.absolutePath
            }
        }
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, OrtSession.SessionOptions())

    actual val supportsInference: Boolean = true

    actual fun predict(sequence: List<FloatArray>): TemporalPrediction {
        val padded = ArrayList<FloatArray>(5)
        if (sequence.size < 5) {
            val paddingCount = 5 - sequence.size
            for (i in 0 until paddingCount) {
                padded.add(FloatArray(384))
            }
        }
        padded.addAll(sequence.takeLast(5))

        val flatArray = FloatArray(5 * 384)
        for (i in 0 until 5) {
            val emb = padded[i]
            val actualSize = minOf(emb.size, 384)
            System.arraycopy(emb, 0, flatArray, i * 384, actualSize)
        }

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(flatArray),
            longArrayOf(1, 5, 384)
        )

        val logits: FloatArray
        inputTensor.use { tensor ->
            val outputs = session.run(mapOf("embedding_sequence" to tensor))
            outputs.use { result ->
                val outputValue = result[0].value
                logits = when (outputValue) {
                    is Array<*> -> {
                        @Suppress("UNCHECKED_CAST")
                        (outputValue as Array<FloatArray>)[0]
                    }
                    else -> error("Unexpected ONNX output type for temporal classifier logits")
                }
            }
        }

        val probs = softmax(logits)
        val pairProb = probs.getOrElse(1) { 0f }
        val crescendoProb = probs.getOrElse(2) { 0f }

        return TemporalPrediction(
            crescendoProbability = crescendoProb,
            pairProbability = pairProb
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return FloatArray(0)
        val max = logits.maxOrNull() ?: 0f
        val exps = logits.map { kotlin.math.exp((it - max).toDouble()).toFloat() }
        val sum = exps.sum().takeIf { it > 0f } ?: return FloatArray(logits.size)
        return FloatArray(logits.size) { i -> exps[i] / sum }
    }

    actual override fun close() {
        session.close()
        env.close()
    }
}
