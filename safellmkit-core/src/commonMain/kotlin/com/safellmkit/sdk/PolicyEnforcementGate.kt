package com.safellmkit.sdk

import com.safellmkit.core.engine.GuardrailEngine
import com.safellmkit.core.model.ConversationTurn
import com.safellmkit.core.model.GuardrailAction
import com.safellmkit.core.model.GuardrailResult
import com.safellmkit.sdk.provider.LlmProvider
import com.safellmkit.sdk.provider.ProviderResponse

/**
 * Single enforcement gate for all provider calls.
 * Input is inspected before the provider; output is inspected before returning to the caller.
 */
object PolicyEnforcementGate {

    suspend fun execute(
        engine: GuardrailEngine,
        provider: LlmProvider,
        request: ChatRequest,
        policy: SdkPolicy
    ): ChatResponse {
        val turnIndex = engine.memory.getConversationState(request.sessionId)?.totalTurns ?: request.turnIndex

        val inputTurn = ConversationTurn(
            sessionId = request.sessionId,
            turnIndex = turnIndex,
            role = "user",
            content = request.prompt,
            additionalContext = request.additionalContext.map { it.content },
            userId = request.userId
        )

        val inputResult = inspectSafely(engine, inputTurn, policy, stage = "input")
            ?: return failClosedResponse(
                stage = "input",
                message = "Guardrail inspection failed before provider call"
            )

        val inputBlock = shouldBlockInput(inputResult, policy)
        if (inputBlock != null) {
            return toBlockedResponse(inputResult, inputBlock, stage = "input", providerCalled = false)
        }

        val promptForProvider = resolveSafePrompt(inputResult, request.prompt)

        // Inspect additionalContext blocks (Fix 4)
        val safeContextBlocks = mutableListOf<ContextBlock>()
        for (block in request.additionalContext) {
            val contextTurn = ConversationTurn(
                sessionId = request.sessionId,
                turnIndex = turnIndex,
                role = "user",
                content = block.content,
                additionalContext = emptyList(),
                userId = request.userId
            )
            val contextResult = inspectSafely(engine, contextTurn, policy, stage = "input_context")
                ?: return failClosedResponse(
                    stage = "input_context",
                    message = "Guardrail inspection failed on context block before provider call"
                )
            val contextBlockMsg = shouldBlockInput(contextResult, policy)
            if (contextBlockMsg != null) {
                return toBlockedResponse(contextResult, contextBlockMsg, stage = "input_context", providerCalled = false)
            }
            val safeContent = resolveSafePrompt(contextResult, block.content)
            safeContextBlocks.add(block.copy(content = safeContent))
        }

        val assembledPrompt = if (safeContextBlocks.isEmpty()) {
            promptForProvider
        } else {
            promptForProvider + "\n\nContext:\n" + safeContextBlocks.joinToString("\n") { 
                "[Source: ${it.source ?: "unknown"}]: ${it.content}" 
            }
        }

        val providerResponse: ProviderResponse
        try {
            providerResponse = provider.generate(
                request.copy(prompt = assembledPrompt, turnIndex = turnIndex, additionalContext = safeContextBlocks)
            )
        } catch (e: Exception) {
            return if (policy.failClosedOnError) {
                failClosedResponse(
                    stage = "provider",
                    message = "Provider call failed: ${e.message}",
                    diagnostics = mapOf("exception" to e::class.simpleName)
                )
            } else {
                throw e
            }
        }

        val outputTurn = ConversationTurn(
            sessionId = request.sessionId,
            turnIndex = turnIndex + 1,
            role = "assistant",
            content = providerResponse.text,
            additionalContext = emptyList(),
            userId = request.userId
        )

        val outputResult = inspectSafely(engine, outputTurn, policy, stage = "output")
            ?: return failClosedResponse(
                stage = "output",
                message = "Guardrail inspection failed after provider call",
                diagnostics = mapOf("providerCalled" to true)
            )

        if (outputResult.action == GuardrailAction.BLOCK) {
            return toBlockedResponse(
                outputResult,
                "Model output blocked by guardrails",
                stage = "output",
                providerCalled = true
            )
        }

        val safeOutput = when (outputResult.action) {
            GuardrailAction.REDACT, GuardrailAction.SANITIZE -> outputResult.safeText
            else -> providerResponse.text
        }

        return ChatResponse(
            blocked = false,
            action = GuardrailAction.ALLOW,
            message = null,
            response = safeOutput,
            riskScore = outputResult.riskScore,
            reasons = outputResult.reasons,
            diagnostics = mapOf(
                "inputAction" to inputResult.action.name,
                "outputAction" to outputResult.action.name,
                "providerCalled" to true
            )
        )
    }

    private suspend fun inspectSafely(
        engine: GuardrailEngine,
        turn: ConversationTurn,
        policy: SdkPolicy,
        stage: String
    ): GuardrailResult? {
        return try {
            engine.inspect(turn)
        } catch (e: Exception) {
            if (policy.failClosedOnError) null else throw e
        }
    }

    private fun shouldBlockInput(result: GuardrailResult, policy: SdkPolicy): String? {
        return when (result.action) {
            GuardrailAction.BLOCK -> result.messageToUser.ifBlank { "Input blocked by guardrails" }
            GuardrailAction.WARN -> if (policy.blockOnWarn) {
                result.messageToUser.ifBlank { "Input warning escalated to block" }
            } else null
            else -> null
        }
    }

    private fun resolveSafePrompt(result: GuardrailResult, original: String): String {
        return when (result.action) {
            GuardrailAction.REDACT, GuardrailAction.SANITIZE -> result.safeText ?: original
            else -> result.safeText ?: original
        }
    }

    private fun toBlockedResponse(
        result: GuardrailResult,
        message: String,
        stage: String,
        providerCalled: Boolean
    ): ChatResponse {
        return ChatResponse(
            blocked = true,
            action = GuardrailAction.BLOCK,
            message = message,
            response = null,
            riskScore = result.riskScore,
            reasons = result.reasons,
            diagnostics = mapOf(
                "stage" to stage,
                "providerCalled" to providerCalled,
                "guardrailAction" to result.action.name,
                "userStatus" to result.userState?.conversationStatus?.name
            )
        )
    }

    private fun failClosedResponse(
        stage: String,
        message: String,
        diagnostics: Map<String, Any?> = emptyMap()
    ): ChatResponse {
        return ChatResponse(
            blocked = true,
            action = GuardrailAction.BLOCK,
            message = message,
            response = null,
            riskScore = 100f,
            reasons = listOf(message),
            diagnostics = diagnostics + mapOf("stage" to stage, "failClosed" to true, "providerCalled" to false)
        )
    }
}
