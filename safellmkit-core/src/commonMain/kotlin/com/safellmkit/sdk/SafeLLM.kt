package com.safellmkit.sdk

import com.safellmkit.sdk.provider.OpenRouterProvider

/**
 * Public entry point for SafeLLMKit.
 *
 * ```kotlin
 * val client = SafeLLM.builder()
 *     .provider(OpenRouterProvider(apiKey = "...", model = "..."))
 *     .build()
 *
 * val response = client.chat("Hello")
 * ```
 */
object SafeLLM {

    fun builder(): SafeLLMBuilder = SafeLLMBuilder()

    fun createDefault(apiKey: String, model: String): SafeLLMClient {
        return builder()
            .provider(OpenRouterProvider(apiKey = apiKey, model = model))
            .build()
    }
}
