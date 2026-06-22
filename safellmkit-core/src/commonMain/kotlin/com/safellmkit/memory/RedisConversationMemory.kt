package com.safellmkit.memory

import com.safellmkit.core.engine.ConversationMemoryGateway
import com.safellmkit.core.model.ConversationTurn
import com.safellmkit.core.model.ConversationState
import com.safellmkit.core.model.ConversationSnapshot
import io.ktor.network.selector.SelectorManager
import io.ktor.network.sockets.aSocket
import io.ktor.network.sockets.openReadChannel
import io.ktor.network.sockets.openWriteChannel
import io.ktor.utils.io.ByteReadChannel
import io.ktor.utils.io.ByteWriteChannel
import io.ktor.utils.io.readFully
import io.ktor.utils.io.writeFully
import io.ktor.utils.io.writeStringUtf8
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable

@Serializable
data class RedisSessionState(
    val turns: List<ConversationTurn> = emptyList(),
    val embeddings: List<List<Float>> = emptyList(),
    val state: ConversationState? = null
)

class RedisConversationMemory(
    val host: String = "127.0.0.1",
    val port: Int = 6379,
    val fallback: ConversationMemoryGateway = InMemoryConversationMemory()
) : ConversationMemoryGateway {

    private val selectorManager = SelectorManager(Dispatchers.Default)
    private val json = kotlinx.serialization.json.Json { ignoreUnknownKeys = true }

    private suspend fun <T> runRedis(block: suspend (ByteWriteChannel, ByteReadChannel) -> T): T {
        return withContext(Dispatchers.Default) {
            val socket = aSocket(selectorManager).tcp().connect(host, port)
            try {
                val writeChannel = socket.openWriteChannel(autoFlush = true)
                val readChannel = socket.openReadChannel()
                block(writeChannel, readChannel)
            } finally {
                socket.close()
            }
        }
    }

    private suspend fun getSessionState(sessionId: String): RedisSessionState {
        return runRedis { write, read ->
            write.writeRespCommand(listOf("GET", "session:$sessionId:state"))
            val response = read.readRespResponse()
            val bytes = response as? ByteArray
            if (bytes != null) {
                json.decodeFromString(RedisSessionState.serializer(), bytes.decodeToString())
            } else {
                RedisSessionState()
            }
        }
    }

    private suspend fun saveSessionState(sessionId: String, sessionState: RedisSessionState) {
        val jsonStr = json.encodeToString(RedisSessionState.serializer(), sessionState)
        runRedis { write, read ->
            write.writeRespCommand(listOf("SET", "session:$sessionId:state", jsonStr))
            read.readRespResponse() // Consume response OK
        }
    }

    private suspend fun <T> executeOrFallback(fallbackBlock: suspend () -> T, redisBlock: suspend () -> T): T {
        return try {
            redisBlock()
        } catch (e: Throwable) {
            fallbackBlock()
        }
    }

    override suspend fun append(turn: ConversationTurn, embedding: FloatArray) {
        executeOrFallback({ fallback.append(turn, embedding) }) {
            // 1. Persist the state engine json blob
            val state = getSessionState(turn.sessionId)
            val updatedTurns = state.turns + turn
            val updatedEmbeddings = state.embeddings + listOf(embedding.toList())
            saveSessionState(turn.sessionId, state.copy(turns = updatedTurns, embeddings = updatedEmbeddings))

            // 2. Synchronously write vector indices to global index spaces via Redis 8 native VADD
            // Syntax: VADD key VALUES dim element
            val vectorString = embedding.joinToString(" ") { it.toString() }
            val vectorIndexKey = "global:attack_vectors"
            val elementId = "turn:${turn.sessionId}:${turn.timestamp}"

            runRedis { write, read ->
                write.writeRespCommand(listOf(
                    "VADD", 
                    vectorIndexKey, 
                    "VALUES", 
                    embedding.size.toString(), 
                    vectorString
                ))
                read.readRespResponse() // Consume VADD response integer

                // Attach payload string descriptor metadata using VSETATTR
                write.writeRespCommand(listOf(
                    "VSETATTR",
                    vectorIndexKey,
                    elementId,
                    "prompt_id:${turn.sessionId}"
                ))
                read.readRespResponse() // Consume VSETATTR response OK
            }
        }
    }

    override suspend fun queryVectorSimilarity(embedding: FloatArray): Float {
        return executeOrFallback({ 0f }) {
            // Redis 8 parses float vectors directly as whitespace-delimited numerical text strings
            val vectorString = embedding.joinToString(" ") { it.toString() }
            val vectorIndexKey = "global:attack_vectors"

            runRedis { write, read ->
                // Native Syntax: VSIM key VALUES dim vector COUNT n
                write.writeRespCommand(listOf(
                    "VSIM",
                    vectorIndexKey,
                    "VALUES",
                    embedding.size.toString(),
                    vectorString,
                    "COUNT",
                    "1"
                ))
                val response = read.readRespResponse()
                parseRedis8VectorSearchResponse(response)
            }
        }
    }

    private fun parseRedis8VectorSearchResponse(resp: Any?): Float {
        // Redis 8 VSIM returns an array payload containing match strings and explicit similarity score floats
        if (resp !is List<*>) return 0f
        if (resp.isEmpty()) return 0f
        
        // Iterating across returning pairs [elementId, similarityScore]
        // Since we query COUNT 1, check the secondary property directly if available
        val similarityValue = resp.getOrNull(1) ?: return 0f
        return when (similarityValue) {
            is ByteArray -> similarityValue.decodeToString().toFloatOrNull() ?: 0f
            is String -> similarityValue.toFloatOrNull() ?: 0f
            is Long -> similarityValue.toFloat()
            else -> 0f
        }
    }

    // --- Keep interface overrides for fallback tracking ---

    override suspend fun rollingCentroid(sessionId: String, historyWindow: Int): FloatArray? {
        return executeOrFallback({ fallback.rollingCentroid(sessionId, historyWindow) }) {
            val state = getSessionState(sessionId)
            val embs = state.embeddings.takeLast(historyWindow)
            if (embs.isEmpty()) return@executeOrFallback null
            val dim = embs[0].size
            val centroid = FloatArray(dim)
            for (emb in embs) {
                for (i in 0 until dim) centroid[i] += emb[i]
            }
            for (i in 0 until dim) centroid[i] /= embs.size.toFloat()
            centroid
        }
    }

    override suspend fun recentTurns(sessionId: String, limit: Int): List<ConversationTurn> {
        return executeOrFallback({ fallback.recentTurns(sessionId, limit) }) {
            val state = getSessionState(sessionId)
            state.turns.takeLast(limit)
        }
    }

    override suspend fun recentEmbeddings(sessionId: String, limit: Int): List<FloatArray> {
        return executeOrFallback({ fallback.recentEmbeddings(sessionId, limit) }) {
            val state = getSessionState(sessionId)
            state.embeddings.takeLast(limit).map { it.toFloatArray() }
        }
    }

    override suspend fun getConversationState(sessionId: String): ConversationState? {
        return executeOrFallback({ fallback.getConversationState(sessionId) }) {
            val state = getSessionState(sessionId)
            state.state
        }
    }

    override suspend fun updateConversationState(state: ConversationState) {
        executeOrFallback({ fallback.updateConversationState(state) }) {
            val sessionState = getSessionState(state.sessionId)
            saveSessionState(state.sessionId, sessionState.copy(state = state))
        }
    }

    override suspend fun getSnapshot(sessionId: String, limit: Int): ConversationSnapshot? {
        return executeOrFallback({ fallback.getSnapshot(sessionId, limit) }) {
            val sessionState = getSessionState(sessionId)
            val sessionTurns = sessionState.turns.takeLast(limit)
            val state = sessionState.state ?: return@executeOrFallback null
            ConversationSnapshot(
                sessionId = sessionId,
                turns = sessionTurns,
                state = state
            )
        }
    }

    // --- Low level RESP Protocol Socket Writers and Readers ---

    private suspend fun ByteWriteChannel.writeRespCommand(args: List<Any>) {
        writeStringUtf8("*${args.size}\r\n")
        for (arg in args) {
            when (arg) {
                is String -> {
                    val bytes = arg.encodeToByteArray()
                    writeStringUtf8("$${bytes.size}\r\n")
                    writeFully(bytes)
                    writeStringUtf8("\r\n")
                }
                is ByteArray -> {
                    writeStringUtf8("$${arg.size}\r\n")
                    writeFully(arg)
                    writeStringUtf8("\r\n")
                }
                else -> {
                    val str = arg.toString()
                    val bytes = str.encodeToByteArray()
                    writeStringUtf8("$${bytes.size}\r\n")
                    writeFully(bytes)
                    writeStringUtf8("\r\n")
                }
            }
        }
        flush()
    }

    private suspend fun ByteReadChannel.readRespResponse(): Any? {
        val line = readUTF8Line(Int.MAX_VALUE) ?: throw Exception("Connection closed by server")
        if (line.isEmpty()) return null
        val type = line[0]
        val value = line.substring(1)
        return when (type) {
            '+' -> value
            '-' -> throw Exception("Redis error: $value")
            ':' -> value.toLong()
            '$' -> {
                val length = value.toInt()
                if (length == -1) {
                    null
                } else {
                    val bytes = ByteArray(length)
                    readFully(bytes)
                    readUTF8Line(Int.MAX_VALUE) // Read trailing \r\n
                    bytes
                }
            }
            '*' -> {
                val length = value.toInt()
                if (length == -1) {
                    null
                } else {
                    val list = mutableListOf<Any?>()
                    for (i in 0 until length) {
                        list.add(readRespResponse())
                    }
                    list
                }
            }
            else -> throw Exception("Unknown RESP type: $type")
        }
    }
}