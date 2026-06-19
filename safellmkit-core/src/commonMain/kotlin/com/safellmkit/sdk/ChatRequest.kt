package com.safellmkit.sdk

data class ChatRequest(
    val userId: String,
    val sessionId: String,
    val prompt: String,
    val turnIndex: Int = 0
)
