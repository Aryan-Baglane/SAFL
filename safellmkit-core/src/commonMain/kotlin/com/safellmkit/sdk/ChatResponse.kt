package com.safellmkit.sdk

import com.safellmkit.core.model.GuardrailAction

data class ChatResponse(
    val blocked: Boolean,
    val action: GuardrailAction,
    val message: String?,
    val response: String?,
    val riskScore: Float,
    val reasons: List<String>,
    val diagnostics: Map<String, Any?> = emptyMap()
)
