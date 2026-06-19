package com.safellmkit.sdk.provider

data class ProviderResponse(
    val text: String,
    val model: String? = null,
    val raw: Map<String, Any?> = emptyMap()
)
