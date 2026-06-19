package com.safellmkit.sdk

import com.safellmkit.core.engine.GuardrailEngine
import com.safellmkit.sdk.provider.LlmProvider

/**
 * High-level SDK client. All chat traffic routes through [PolicyEnforcementGate].
 */
@OptIn(kotlin.ExperimentalStdlibApi::class)
class SafeLLMClient internal constructor(
    private val engine: GuardrailEngine,
    private val provider: LlmProvider,
    private val policy: SdkPolicy,
    private val defaultUserId: String,
    private val defaultSessionId: String
) : AutoCloseable {

    suspend fun chat(prompt: String): ChatResponse =
        chat(userId = defaultUserId, sessionId = defaultSessionId, prompt = prompt)

    suspend fun chat(userId: String, sessionId: String, prompt: String): ChatResponse {
        val request = ChatRequest(
            userId = userId,
            sessionId = sessionId,
            prompt = prompt
        )
        return PolicyEnforcementGate.execute(
            engine = engine,
            provider = provider,
            request = request,
            policy = policy
        )
    }

    suspend fun chat(request: ChatRequest): ChatResponse {
        return PolicyEnforcementGate.execute(
            engine = engine,
            provider = provider,
            request = request,
            policy = policy
        )
    }

    override fun close() {
        engine.close()
    }
}
