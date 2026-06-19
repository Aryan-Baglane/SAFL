package com.safellmkit.sdk

import com.safellmkit.core.engine.GuardrailEngine
import com.safellmkit.core.policy.GuardrailsPolicies
import com.safellmkit.core.policy.GuardrailsPolicy
import com.safellmkit.sdk.provider.LlmProvider

class SafeLLMBuilder {
    private var provider: LlmProvider? = null
    private var policy: SdkPolicy = SdkPolicy()
    private var engine: GuardrailEngine? = null
    private var defaultUserId: String = "default-user"
    private var defaultSessionId: String = "default-session"

    fun provider(provider: LlmProvider): SafeLLMBuilder = apply {
        this.provider = provider
    }

    fun policy(policy: SdkPolicy): SafeLLMBuilder = apply {
        this.policy = policy
    }

    fun guardrailsPolicy(guardrailsPolicy: GuardrailsPolicy): SafeLLMBuilder = apply {
        this.policy = policy.copy(guardrailsPolicy = guardrailsPolicy)
    }

    /** Enables fail-closed enforcement. Does not override a custom [guardrailsPolicy]. */
    fun strict(): SafeLLMBuilder = apply {
        policy = policy.copy(
            guardrailsPolicy = policy.guardrailsPolicy,
            failClosedOnError = true,
            blockOnWarn = false
        )
    }

    fun failClosed(enabled: Boolean = true): SafeLLMBuilder = apply {
        policy = policy.copy(failClosedOnError = enabled)
    }

    fun blockOnWarn(enabled: Boolean = true): SafeLLMBuilder = apply {
        policy = policy.copy(blockOnWarn = enabled)
    }

    fun defaultUserId(userId: String): SafeLLMBuilder = apply {
        defaultUserId = userId
    }

    fun defaultSessionId(sessionId: String): SafeLLMBuilder = apply {
        defaultSessionId = sessionId
    }

    /** Advanced: inject a pre-configured engine (testing or custom memory). */
    fun engine(engine: GuardrailEngine): SafeLLMBuilder = apply {
        this.engine = engine
    }

    fun build(): SafeLLMClient {
        val selectedProvider = provider
            ?: error("SafeLLM.builder() requires .provider(...)")
        val guardrailEngine = engine ?: GuardrailEngine(policy = policy.guardrailsPolicy)
        return SafeLLMClient(
            engine = guardrailEngine,
            provider = selectedProvider,
            policy = policy,
            defaultUserId = defaultUserId,
            defaultSessionId = defaultSessionId
        )
    }
}
