package com.safellmkit.core.engine

interface LlmValidator {
    suspend fun validate(prompt: String): Float?
}

class GeminiValidator : LlmValidator {
    override suspend fun validate(prompt: String): Float? = null
}

class OpenAIValidator : LlmValidator {
    override suspend fun validate(prompt: String): Float? = null
}

class ClaudeValidator : LlmValidator {
    override suspend fun validate(prompt: String): Float? = null
}

class OllamaValidator : LlmValidator {
    override suspend fun validate(prompt: String): Float? = null
}
