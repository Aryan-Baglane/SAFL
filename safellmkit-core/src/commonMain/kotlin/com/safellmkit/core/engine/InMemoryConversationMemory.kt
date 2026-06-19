package com.safellmkit.core.engine

import com.safellmkit.core.model.ConversationSnapshot
import com.safellmkit.core.model.ConversationState
import com.safellmkit.core.model.ConversationTurn

@Deprecated(
    message = "Use com.safellmkit.memory.InMemoryConversationMemory instead",
    replaceWith = ReplaceWith("InMemoryConversationMemory", "com.safellmkit.memory.InMemoryConversationMemory")
)
class InMemoryConversationMemory : ConversationMemoryGateway {
    private val delegate = com.safellmkit.memory.InMemoryConversationMemory()

    override suspend fun append(turn: ConversationTurn, embedding: FloatArray) = delegate.append(turn, embedding)
    override suspend fun rollingCentroid(sessionId: String, historyWindow: Int): FloatArray? = delegate.rollingCentroid(sessionId, historyWindow)
    override suspend fun recentTurns(sessionId: String, limit: Int): List<ConversationTurn> = delegate.recentTurns(sessionId, limit)
    override suspend fun getConversationState(sessionId: String): ConversationState? = delegate.getConversationState(sessionId)
    override suspend fun updateConversationState(state: ConversationState) = delegate.updateConversationState(state)
    override suspend fun getSnapshot(sessionId: String, limit: Int): ConversationSnapshot? = delegate.getSnapshot(sessionId, limit)
    override suspend fun queryVectorSimilarity(embedding: FloatArray): Float = delegate.queryVectorSimilarity(embedding)
    override suspend fun recentEmbeddings(sessionId: String, limit: Int): List<FloatArray> = delegate.recentEmbeddings(sessionId, limit)
}
