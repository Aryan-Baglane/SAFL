package com.safellmkit.sdk

import kotlinx.serialization.Serializable

@Serializable
data class ContextBlock(
    val content: String,
    val source: String? = null
)
