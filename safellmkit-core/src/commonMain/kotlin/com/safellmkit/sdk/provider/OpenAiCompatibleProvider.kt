package com.safellmkit.sdk.provider

import com.safellmkit.sdk.ChatRequest
import com.safellmkit.sdk.http.createHttpClient
import io.ktor.client.call.body
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
private data class ChatMessage(
    val role: String,
    val content: String
)

@Serializable
private data class ChatCompletionRequest(
    val model: String,
    val messages: List<ChatMessage>
)

@Serializable
private data class ChatCompletionChoice(
    val message: ChatMessage? = null
)

@Serializable
private data class ChatCompletionResponse(
    val choices: List<ChatCompletionChoice> = emptyList(),
    val model: String? = null
)

/**
 * Shared OpenAI-compatible chat completions client for OpenRouter and OpenAI.
 */
abstract class OpenAiCompatibleProvider(
    private val apiKey: String,
    private val model: String,
    private val baseUrl: String,
    private val extraHeaders: Map<String, String> = emptyMap()
) : LlmProvider {

    private val client = createHttpClient().config {
        install(ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true })
        }
    }

    override suspend fun generate(request: ChatRequest): ProviderResponse {
        val httpResponse = client.post("$baseUrl/chat/completions") {
            contentType(ContentType.Application.Json)
            header("Authorization", "Bearer $apiKey")
            extraHeaders.forEach { (key, value) -> header(key, value) }
            setBody(
                ChatCompletionRequest(
                    model = model,
                    messages = listOf(ChatMessage(role = "user", content = request.prompt))
                )
            )
        }
        val body: ChatCompletionResponse = httpResponse.body()
        val text = body.choices.firstOrNull()?.message?.content
            ?: error("Provider returned no choices")
        return ProviderResponse(text = text, model = body.model ?: model)
    }
}

class OpenRouterProvider(
    apiKey: String,
    model: String,
    extraHeaders: Map<String, String> = emptyMap()
) : OpenAiCompatibleProvider(
    apiKey = apiKey,
    model = model,
    baseUrl = "https://openrouter.ai/api/v1",
    extraHeaders = extraHeaders
)

class OpenAIProvider(
    apiKey: String,
    model: String
) : OpenAiCompatibleProvider(
    apiKey = apiKey,
    model = model,
    baseUrl = "https://api.openai.com/v1"
)
