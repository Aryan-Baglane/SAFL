package com.safellmkit.sdk.provider

import com.safellmkit.sdk.ChatRequest

/**
 * Raw LLM backend. Must only be invoked through [com.safellmkit.sdk.PolicyEnforcementGate].
 */
interface LlmProvider {
    suspend fun generate(request: ChatRequest): ProviderResponse
}
