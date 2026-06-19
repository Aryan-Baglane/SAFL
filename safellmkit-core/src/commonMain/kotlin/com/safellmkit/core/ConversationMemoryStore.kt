package com.safellmkit.core.engine

import com.safellmkit.core.model.ConversationTurn
import kotlin.math.sqrt



class HashingEmbeddingProvider : EmbeddingProvider {
    override fun embed(text: String, dimensions: Int): FloatArray {
        val dim = if (dimensions <= 0) 128 else dimensions
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

        return normalize(vector)
    }

    private fun normalize(vec: FloatArray): FloatArray {
        var sumSq = 0.0
        for (v in vec) sumSq += (v * v).toDouble()
        val norm = sqrt(sumSq).toFloat()
        if (norm == 0f) return vec
        for (i in vec.indices) vec[i] /= norm
        return vec
    }
}

fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    val n = minOf(a.size, b.size)
    var dot = 0f
    var na = 0f
    var nb = 0f
    for (i in 0 until n) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    if (na == 0f || nb == 0f) return 0f
    return (dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb))).coerceIn(-1f, 1f)
}

class ConversationMemoryStore(
    private val maxTurns: Int = 32
) {
    private val turnsByConversation = mutableMapOf<String, ArrayDeque<ConversationTurn>>()
    private val embeddingsByConversation = mutableMapOf<String, ArrayDeque<FloatArray>>()

    fun append(turn: ConversationTurn, embedding: FloatArray) {
        val turns = turnsByConversation.getOrPut(turn.sessionId) { ArrayDeque() }
        val embeddings = embeddingsByConversation.getOrPut(turn.sessionId) { ArrayDeque() }

        turns.addLast(turn)
        embeddings.addLast(embedding)

        while (turns.size > maxTurns) turns.removeFirst()
        while (embeddings.size > maxTurns) embeddings.removeFirst()
    }

    fun recentTurns(conversationId: String, limit: Int): List<ConversationTurn> {
        return turnsByConversation[conversationId]
            ?.takeLast(limit)
            ?.toList()
            ?: emptyList()
    }

    fun rollingCentroid(conversationId: String, limit: Int = 8): FloatArray? {
        val embeddings = embeddingsByConversation[conversationId]?.takeLast(limit)?.toList() ?: return null
        if (embeddings.isEmpty()) return null

        val dim = embeddings[0].size
        val centroid = FloatArray(dim)

        for (emb in embeddings) {
            for (i in 0 until dim) centroid[i] += emb[i]
        }

        for (i in 0 until dim) centroid[i] /= embeddings.size.toFloat()
        return centroid
    }
}