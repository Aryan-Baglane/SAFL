package com.safellmkit.sdk.testsupport

import com.safellmkit.sdk.ChatRequest
import com.safellmkit.sdk.provider.LlmProvider
import com.safellmkit.sdk.provider.ProviderResponse
import java.util.concurrent.atomic.AtomicInteger

class MockLlmProvider(
    private val responseText: String = "Hello from the model"
) : LlmProvider {

    val callCount = AtomicInteger(0)
    val lastRequest = java.util.concurrent.atomic.AtomicReference<ChatRequest?>(null)

    override suspend fun generate(request: ChatRequest): ProviderResponse {
        callCount.incrementAndGet()
        lastRequest.set(request)
        return ProviderResponse(text = responseText)
    }
}
