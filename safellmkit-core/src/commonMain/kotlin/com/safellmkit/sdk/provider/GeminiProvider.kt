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
private data class GeminiPart(val text: String)

@Serializable
private data class GeminiContent(val parts: List<GeminiPart>)

@Serializable
private data class GeminiRequest(val contents: List<GeminiContent>)

@Serializable
private data class GeminiCandidate(val content: GeminiContent? = null)

@Serializable
private data class GeminiResponse(val candidates: List<GeminiCandidate> = emptyList())

class GeminiProvider(
    private val apiKey: String,
    private val model: String
) : LlmProvider {

    private val client = createHttpClient().config {
        install(ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true })
        }
    }

    override suspend fun generate(request: ChatRequest): ProviderResponse {
        val url =
            "https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$apiKey"
        val httpResponse = client.post(url) {
            contentType(ContentType.Application.Json)
            setBody(
                GeminiRequest(
                    contents = listOf(
                        GeminiContent(parts = listOf(GeminiPart(text = request.prompt)))
                    )
                )
            )
        }
        val body: GeminiResponse = httpResponse.body()
        val text = body.candidates.firstOrNull()?.content?.parts?.firstOrNull()?.text
            ?: error("Gemini returned no candidates")
        return ProviderResponse(text = text, model = model)
    }
}
