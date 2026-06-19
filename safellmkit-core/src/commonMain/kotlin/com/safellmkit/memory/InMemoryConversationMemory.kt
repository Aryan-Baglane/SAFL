package com.safellmkit.memory

import com.safellmkit.core.engine.ConversationMemoryGateway
import com.safellmkit.core.model.ConversationTurn
import com.safellmkit.core.model.ConversationState
import com.safellmkit.core.model.ConversationSnapshot

class InMemoryConversationMemory : ConversationMemoryGateway {
    private val turns = mutableMapOf<String, MutableList<ConversationTurn>>()
    private val embeddings = mutableMapOf<String, MutableList<FloatArray>>()
    private val states = mutableMapOf<String, ConversationState>()

    override suspend fun append(turn: ConversationTurn, embedding: FloatArray) {
        turns.getOrPut(turn.sessionId) { mutableListOf() }.add(turn)
        embeddings.getOrPut(turn.sessionId) { mutableListOf() }.add(embedding)
    }

    override suspend fun rollingCentroid(sessionId: String, historyWindow: Int): FloatArray? {
        val embs = embeddings[sessionId]?.takeLast(historyWindow) ?: return null
        if (embs.isEmpty()) return null
        val dim = embs[0].size
        val centroid = FloatArray(dim)
        for (emb in embs) {
            for (i in 0 until dim) centroid[i] += emb[i]
        }
        for (i in 0 until dim) centroid[i] /= embs.size.toFloat()
        return centroid
    }

    override suspend fun recentTurns(sessionId: String, limit: Int): List<ConversationTurn> {
        return turns[sessionId]?.takeLast(limit) ?: emptyList()
    }

    override suspend fun recentEmbeddings(sessionId: String, limit: Int): List<FloatArray> {
        return embeddings[sessionId]?.takeLast(limit) ?: emptyList()
    }

    override suspend fun getConversationState(sessionId: String): ConversationState? {
        return states[sessionId]
    }

    override suspend fun updateConversationState(state: ConversationState) {
        states[state.sessionId] = state
    }

    override suspend fun getSnapshot(sessionId: String, limit: Int): ConversationSnapshot? {
        val sessionTurns = turns[sessionId]?.takeLast(limit) ?: emptyList()
        val state = states[sessionId] ?: return null
        return ConversationSnapshot(
            sessionId = sessionId,
            turns = sessionTurns,
            state = state
        )
    }
}
