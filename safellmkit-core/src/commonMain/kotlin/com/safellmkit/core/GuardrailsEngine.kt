package com.safellmkit.core

import com.safellmkit.core.engine.GuardrailEngine
import com.safellmkit.core.model.ConversationTurn
import com.safellmkit.core.policy.GuardrailsPolicy
import kotlinx.coroutines.runBlocking

@Deprecated(
    message = "Use com.safellmkit.sdk.SafeLLMClient for enforced chat. For direct inspection use GuardrailEngine.",
    replaceWith = ReplaceWith("SafeLLMClient", "com.safellmkit.sdk.SafeLLMClient")
)
class GuardrailsEngine(
    val policy: GuardrailsPolicy = com.safellmkit.core.policy.GuardrailsPolicies.strict()
) {
    private val delegate = GuardrailEngine(policy = policy)

    fun validateInput(input: String): GuardrailResult {
        return runBlocking {
            delegate.inspect(ConversationTurn(sessionId = "default", turnIndex = 0, content = input))
        }
    }

    fun validateOutput(output: String): GuardrailResult {
        return runBlocking {
            delegate.inspect(ConversationTurn(sessionId = "default", turnIndex = 0, role = "assistant", content = output))
        }
    }
}