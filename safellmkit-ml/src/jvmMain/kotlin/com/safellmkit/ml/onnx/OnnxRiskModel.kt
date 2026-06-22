package com.safellmkit.ml.onnx

import com.safellmkit.ml.model.AttackTaxonomy
import com.safellmkit.ml.model.RiskClassifier
import com.safellmkit.ml.model.RiskPrediction
import com.safellmkit.ml.tokenizer.TextTokenizer
import com.safellmkit.ml.tokenizer.BasicWordTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import kotlin.math.exp

internal object LabelMapLoader {
    val labels: Map<Int, AttackTaxonomy>

    init {
        val stream = LabelMapLoader::class.java.getResourceAsStream("/assets/label_map.json")
            ?: error("label_map.json not found in assets")
        val jsonStr = stream.readBytes().decodeToString()
        val map = mutableMapOf<Int, AttackTaxonomy>()
        val regex = Regex("\"(\\d+)\"\\s*:\\s*\"([^\"]+)\"")
        regex.findAll(jsonStr).forEach { match ->
            val idx = match.groupValues[1].toInt()
            val name = match.groupValues[2]
            val taxonomy = when (name) {
                "SAFE" -> AttackTaxonomy.SAFE
                "PROMPT_INJECTION" -> AttackTaxonomy.PROMPT_INJECTION
                "JAILBREAK" -> AttackTaxonomy.JAILBREAK
                "GCG" -> AttackTaxonomy.GCG
                "PAIR" -> AttackTaxonomy.PAIR
                "CRESCENDO" -> AttackTaxonomy.CRESCENDO
                "ROLEPLAY" -> AttackTaxonomy.ROLEPLAY
                "ENCODING_ATTACK" -> AttackTaxonomy.ENCODING_ATTACK
                "PII" -> AttackTaxonomy.PII
                else -> AttackTaxonomy.UNKNOWN
            }
            map[idx] = taxonomy
        }
        labels = map
    }
}

private fun getResourceAsTempFile(resourcePath: String, prefix: String, suffix: String): java.io.File {
    val stream = LabelMapLoader::class.java.getResourceAsStream(resourcePath)
        ?: error("Resource not found: $resourcePath")
    val tempFile = java.io.File.createTempFile(prefix, suffix)
    tempFile.deleteOnExit()
    tempFile.outputStream().use { out ->
        stream.copyTo(out)
    }
    return tempFile
}

actual class OnnxRiskModel actual constructor(
    modelPath: String,
    private val tokenizer: TextTokenizer
) : RiskClassifier, AutoCloseable {

    actual constructor() : this(
        modelPath = getResourceAsTempFile("/assets/guardrail_model_int8.onnx", "guardrail_model", ".onnx").absolutePath,
        tokenizer = BasicWordTokenizer(
            LabelMapLoader::class.java.getResourceAsStream("/assets/tokenizer/vocab.txt")
                ?.bufferedReader()?.readLines() ?: error("vocab.txt not found")
        )
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, OrtSession.SessionOptions())

    override actual val supportsInference: Boolean = true

    override actual fun predict(text: String, maxLength: Int): RiskPrediction {
        val tokenized = tokenizer.encode(text, maxLength)

        val inputIds = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(tokenized.inputIds),
            longArrayOf(1, tokenized.inputIds.size.toLong())
        )
        val attentionMask = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(tokenized.attentionMask),
            longArrayOf(1, tokenized.attentionMask.size.toLong())
        )

        val logits: FloatArray
        val embedding: FloatArray

        inputIds.use { ids ->
            attentionMask.use { mask ->
                val outputs = session.run(
                    mapOf(
                        "input_ids" to ids,
                        "attention_mask" to mask
                    )
                )
                outputs.use { result ->
                    val logitsAny = result[0].value
                    logits = when (logitsAny) {
                        is Array<*> -> {
                            @Suppress("UNCHECKED_CAST")
                            (logitsAny as Array<FloatArray>)[0]
                        }
                        else -> error("Unexpected ONNX output type for logits")
                    }

                    embedding = if (result.size() > 1) {
                        val embAny = result[1].value
                        when (embAny) {
                            is Array<*> -> {
                                @Suppress("UNCHECKED_CAST")
                                (embAny as Array<FloatArray>)[0]
                            }
                            else -> computeHashEmbedding(text, 384)
                        }
                    } else {
                        computeHashEmbedding(text, 384)
                    }
                }
            }
        }

        val probs = softmax(logits)
        val sortedProbs = probs.sortedDescending()
        val maxProb = sortedProbs.getOrNull(0) ?: 0f
        val secondMaxProb = sortedProbs.getOrNull(1) ?: 0f
        val margin = maxProb - secondMaxProb

        var entropy = 0f
        for (p in probs) {
            if (p > 0f) {
                entropy -= (p * kotlin.math.log2(p.toDouble())).toFloat()
            }
        }

        val uncertain = margin < 0.20f
        val predictedIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val attackClass = if (uncertain && predictedIndex == 0) AttackTaxonomy.UNKNOWN else (LabelMapLoader.labels[predictedIndex] ?: AttackTaxonomy.UNKNOWN)
        val probability = (1f - probs[0]).coerceIn(0f, 1f)

        return RiskPrediction(
            probability = probability,
            attackClass = attackClass,
            embedding = embedding,
            logits = logits,
            uncertain = uncertain,
            margin = margin,
            entropy = entropy
        )
    }

    actual fun predictAttackProbability(text: String, maxLength: Int): Float {
        return predict(text, maxLength).probability
    }

    private fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return FloatArray(0)
        val max = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp((it - max).toDouble()).toFloat() }
        val sum = exps.sum().takeIf { it > 0f } ?: return FloatArray(logits.size)
        return FloatArray(logits.size) { i -> exps[i] / sum }
    }

    private fun computeHashEmbedding(text: String, dim: Int): FloatArray {
        val vector = FloatArray(dim)
        val normalized = text.lowercase()
        val tokens = normalized
            .split(Regex("[^a-z0-9]+"))
            .filter { it.isNotBlank() }

        for (token in tokens) {
            val index = (token.hashCode() and Int.MAX_VALUE) % dim
            vector[index] += 1f
            if (token.length >= 3) {
                for (i in 0..token.length - 3) {
                    val trigram = token.substring(i, i + 3)
                    val triIndex = (trigram.hashCode() and Int.MAX_VALUE) % dim
                    vector[triIndex] += 0.5f
                }
            }
        }

        var sumSq = 0.0
        for (v in vector) sumSq += (v * v).toDouble()
        val norm = kotlin.math.sqrt(sumSq).toFloat()
        if (norm > 0f) {
            for (i in vector.indices) vector[i] /= norm
        }
        return vector
    }

    override actual fun close() {
        session.close()
        env.close()
    }
}