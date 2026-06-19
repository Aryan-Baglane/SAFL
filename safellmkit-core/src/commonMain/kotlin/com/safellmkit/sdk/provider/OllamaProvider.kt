package com.safellmkit.sdk.provider

import com.safellmkit.sdk.ChatRequest
import com.safellmkit.sdk.http.createHttpClient
import io.ktor.client.call.body
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
private data class OllamaMessage(val role: String, val content: String)

@Serializable
private data class OllamaChatRequest(
    val model: String,
    val messages: List<OllamaMessage>,
    val stream: Boolean = false
)

@Serializable
private data class OllamaChatResponse(
    val message: OllamaMessage? = null,
    val model: String? = null
)

class OllamaProvider(
    private val model: String,
    private val baseUrl: String = "http://localhost:11434"
) : LlmProvider {

    private val client = createHttpClient().config {
        install(ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true })
        }
    }

    override suspend fun generate(request: ChatRequest): ProviderResponse {
        val httpResponse = client.post("$baseUrl/api/chat") {
            contentType(ContentType.Application.Json)
            setBody(
                OllamaChatRequest(
                    model = model,
                    messages = listOf(OllamaMessage(role = "user", content = request.prompt))
                )
            )
        }
        val body: OllamaChatResponse = httpResponse.body()
        val text = body.message?.content ?: error("Ollama returned no message")
        return ProviderResponse(text = text, model = body.model ?: model)
    }
}
