package com.safellmkit.core.memory

import com.safellmkit.core.model.UserState
import com.safellmkit.core.model.AttackTaxonomy
import com.safellmkit.core.model.ConversationStatus
import io.ktor.network.selector.SelectorManager
import io.ktor.network.sockets.aSocket
import io.ktor.network.sockets.openReadChannel
import io.ktor.network.sockets.openWriteChannel
import io.ktor.utils.io.ByteReadChannel
import io.ktor.utils.io.ByteWriteChannel
import io.ktor.utils.io.readFully
import io.ktor.utils.io.writeFully
import io.ktor.utils.io.writeStringUtf8
import io.ktor.utils.io.readUTF8Line
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlin.math.sqrt

interface UserMemoryEngine {
    suspend fun getUserState(userId: String): UserState
    suspend fun updateUserState(state: UserState)
    suspend fun appendSession(userId: String, sessionId: String)
    suspend fun recordAttack(userId: String, attackClass: AttackTaxonomy, risk: Float)
    suspend fun recentEmbeddings(userId: String): List<FloatArray>
    suspend fun recentSessions(userId: String): List<String>
    suspend fun appendEmbedding(
        userId: String,
        sessionId: String,
        embedding: FloatArray,
        attackClass: AttackTaxonomy,
        timestampMs: Long
    )
    suspend fun queryVectorSimilarity(userId: String, embedding: FloatArray): Float
}

class InMemoryUserMemoryEngine : UserMemoryEngine {
    private val mutex = Mutex()
    private val userStates = mutableMapOf<String, UserState>()

    // For HNSW fallback
    private val vectorDb = mutableListOf<VectorRecord>()

    private data class VectorRecord(
        val userId: String,
        val sessionId: String,
        val embedding: FloatArray,
        val attackClass: AttackTaxonomy,
        val timestampMs: Long
    )

    override suspend fun getUserState(userId: String): UserState = mutex.withLock {
        userStates.getOrPut(userId) { UserState(userId = userId) }
    }

    override suspend fun updateUserState(state: UserState) {
        mutex.withLock {
            userStates[state.userId] = state
        }
    }

    override suspend fun appendSession(userId: String, sessionId: String) {
        mutex.withLock {
            val current = userStates.getOrPut(userId) { UserState(userId = userId) }
            if (!current.activeSessions.contains(sessionId)) {
                val updated = current.copy(
                    activeSessions = current.activeSessions + sessionId,
                    sessionHistory = current.sessionHistory + sessionId,
                    lastUpdated = current.lastUpdated
                )
                userStates[userId] = updated
            }
        }
    }

    override suspend fun recordAttack(userId: String, attackClass: AttackTaxonomy, risk: Float) {
        mutex.withLock {
            val current = userStates.getOrPut(userId) { UserState(userId = userId) }
            val updated = current.copy(
                attackCount = current.attackCount + 1,
                riskHistory = current.riskHistory + risk,
                taxonomyHistory = current.taxonomyHistory + attackClass
            )
            userStates[userId] = updated
        }
    }

    override suspend fun recentEmbeddings(userId: String): List<FloatArray> = mutex.withLock {
        val current = userStates[userId] ?: return emptyList()
        current.embeddingHistory.map { it.toFloatArray() }
    }

    override suspend fun recentSessions(userId: String): List<String> = mutex.withLock {
        val current = userStates[userId] ?: return emptyList()
        current.activeSessions
    }

    override suspend fun appendEmbedding(
        userId: String,
        sessionId: String,
        embedding: FloatArray,
        attackClass: AttackTaxonomy,
        timestampMs: Long
    ) {
        mutex.withLock {
            val current = userStates.getOrPut(userId) { UserState(userId = userId) }
            val newEmbs = current.embeddingHistory + listOf(embedding.toList())
            val trimmedEmbs = newEmbs.takeLast(100)
            
            val newAttackVectors = if (attackClass != AttackTaxonomy.SAFE) {
                (current.recentAttackVectors + listOf(embedding.toList())).takeLast(20)
            } else {
                current.recentAttackVectors
            }

            userStates[userId] = current.copy(
                embeddingHistory = trimmedEmbs,
                recentAttackVectors = newAttackVectors
            )

            vectorDb.add(VectorRecord(userId, sessionId, embedding, attackClass, timestampMs))
        }
    }

    override suspend fun queryVectorSimilarity(userId: String, embedding: FloatArray): Float = mutex.withLock {
        var maxSim = 0f
        for (record in vectorDb) {
            if (record.attackClass != AttackTaxonomy.SAFE) {
                val sim = cosine(embedding, record.embedding)
                if (sim > maxSim) {
                    maxSim = sim
                }
            }
        }
        maxSim
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
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
        return (dot / (sqrt(na.toDouble()) * sqrt(nb.toDouble()))).toFloat().coerceIn(-1f, 1f)
    }
}

class RedisUserMemoryEngine(
    val host: String = "127.0.0.1",
    val port: Int = 6379,
    val fallback: UserMemoryEngine = InMemoryUserMemoryEngine()
) : UserMemoryEngine {

    private val selectorManager = SelectorManager(Dispatchers.Default)
    private val json = Json { ignoreUnknownKeys = true }

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

    private suspend fun <T> executeOrFallback(fallbackBlock: suspend () -> T, redisBlock: suspend () -> T): T {
        return try {
            redisBlock()
        } catch (e: Throwable) {
            fallbackBlock()
        }
    }

    override suspend fun getUserState(userId: String): UserState {
        return executeOrFallback({ fallback.getUserState(userId) }) {
            runRedis { write, read ->
                write.writeRespCommand(listOf("GET", "user:$userId:state"))
                val response = read.readRespResponse()
                val bytes = response as? ByteArray
                if (bytes != null) {
                    json.decodeFromString(UserState.serializer(), bytes.decodeToString())
                } else {
                    UserState(userId = userId)
                }
            }
        }
    }

    override suspend fun updateUserState(state: UserState) {
        executeOrFallback({ fallback.updateUserState(state) }) {
            val jsonStr = json.encodeToString(UserState.serializer(), state)
            runRedis { write, read ->
                write.writeRespCommand(listOf("SET", "user:${state.userId}:state", jsonStr))
                read.readRespResponse()
            }
        }
    }

    override suspend fun appendSession(userId: String, sessionId: String) {
        executeOrFallback({ fallback.appendSession(userId, sessionId) }) {
            val current = getUserState(userId)
            if (!current.activeSessions.contains(sessionId)) {
                val updated = current.copy(
                    activeSessions = current.activeSessions + sessionId,
                    sessionHistory = current.sessionHistory + sessionId
                )
                updateUserState(updated)
            }
        }
    }

    override suspend fun recordAttack(userId: String, attackClass: AttackTaxonomy, risk: Float) {
        executeOrFallback({ fallback.recordAttack(userId, attackClass, risk) }) {
            val current = getUserState(userId)
            val updated = current.copy(
                attackCount = current.attackCount + 1,
                riskHistory = current.riskHistory + risk,
                taxonomyHistory = current.taxonomyHistory + attackClass
            )
            updateUserState(updated)
        }
    }

    override suspend fun recentEmbeddings(userId: String): List<FloatArray> {
        return executeOrFallback({ fallback.recentEmbeddings(userId) }) {
            val state = getUserState(userId)
            state.embeddingHistory.map { it.toFloatArray() }
        }
    }

    override suspend fun recentSessions(userId: String): List<String> {
        return executeOrFallback({ fallback.recentSessions(userId) }) {
            val state = getUserState(userId)
            state.activeSessions
        }
    }

    override suspend fun appendEmbedding(
        userId: String,
        sessionId: String,
        embedding: FloatArray,
        attackClass: AttackTaxonomy,
        timestampMs: Long
    ) {
        executeOrFallback({
            fallback.appendEmbedding(userId, sessionId, embedding, attackClass, timestampMs)
        }) {
            val current = getUserState(userId)
            val newEmbs = current.embeddingHistory + listOf(embedding.toList())
            val trimmedEmbs = newEmbs.takeLast(100)
            
            val newAttackVectors = if (attackClass != AttackTaxonomy.SAFE) {
                (current.recentAttackVectors + listOf(embedding.toList())).takeLast(20)
            } else {
                current.recentAttackVectors
            }

            val updated = current.copy(
                embeddingHistory = trimmedEmbs,
                recentAttackVectors = newAttackVectors
            )
            updateUserState(updated)

            // Save to Redis Search user_attack_idx
            if (attackClass != AttackTaxonomy.SAFE) {
                val docId = "user_attack:$userId:$timestampMs"
                val vectorBytes = embedding.toByteArray()
                runRedis { write, read ->
                    write.writeRespCommand(listOf(
                        "HSET",
                        docId,
                        "vector", vectorBytes,
                        "attack_class", attackClass.name,
                        "timestamp", timestampMs.toString(),
                        "sessionId", sessionId,
                        "userId", userId
                    ))
                    read.readRespResponse()
                }
            }
        }
    }

    override suspend fun queryVectorSimilarity(userId: String, embedding: FloatArray): Float {
        return executeOrFallback({ fallback.queryVectorSimilarity(userId, embedding) }) {
            val queryVectorBytes = embedding.toByteArray()
            runRedis { write, read ->
                write.writeRespCommand(listOf(
                    "FT.SEARCH",
                    "user_attack_idx",
                    "@userId:{$userId}=>[KNN 1 @vector \$query_vector AS similarity]",
                    "PARAMS",
                    "2",
                    "query_vector",
                    queryVectorBytes,
                    "DIALECT",
                    "2"
                ))
                val response = read.readRespResponse()
                parseVectorSearchResponse(response)
            }
        }
    }

    private fun parseVectorSearchResponse(resp: Any?): Float {
        if (resp !is List<*>) return 0f
        if (resp.isEmpty() || resp[0] as? Long == 0L) return 0f
        val fields = resp.getOrNull(2) as? List<*> ?: return 0f
        for (i in 0 until fields.size - 1 step 2) {
            val key = when (val k = fields[i]) {
                is ByteArray -> k.decodeToString()
                is String -> k
                else -> null
            }
            if (key == "similarity") {
                val value = when (val v = fields[i + 1]) {
                    is ByteArray -> v.decodeToString()
                    is String -> v
                    else -> null
                }
                return value?.toFloatOrNull() ?: 0f
            }
        }
        return 0f
    }

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

    private fun FloatArray.toByteArray(): ByteArray {
        val bytes = ByteArray(this.size * 4)
        for (i in this.indices) {
            val intBits = this[i].toBits()
            bytes[i * 4] = (intBits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((intBits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((intBits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((intBits shr 24) and 0xFF).toByte()
        }
        return bytes
    }
}
