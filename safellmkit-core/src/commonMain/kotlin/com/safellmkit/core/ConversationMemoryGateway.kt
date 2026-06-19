package com.safellmkit.core.engine

import com.safellmkit.core.model.ConversationSnapshot
import com.safellmkit.core.model.ConversationState
import com.safellmkit.core.model.ConversationTurn

interface ConversationMemoryGateway {
    suspend fun append(turn: ConversationTurn, embedding: FloatArray)
    suspend fun rollingCentroid(sessionId: String, historyWindow: Int): FloatArray?
    suspend fun recentTurns(sessionId: String, limit: Int): List<ConversationTurn>

    suspend fun getConversationState(sessionId: String): ConversationState?
    suspend fun updateConversationState(state: ConversationState)
    suspend fun getSnapshot(sessionId: String, limit: Int = 8): ConversationSnapshot?
    suspend fun queryVectorSimilarity(embedding: FloatArray): Float = 0f
    suspend fun recentEmbeddings(sessionId: String, limit: Int): List<FloatArray> = emptyList()
}