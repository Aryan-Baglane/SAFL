package com.safellmkit.core.firewall

import com.safellmkit.core.GuardrailAction
import com.safellmkit.core.GuardrailResult
import com.safellmkit.core.GuardrailsEngine

data class ToolCall(
    val name: String,
    val argumentsJson: String
)

class ToolFirewall(
    private val allowedTools: Set<String>,
    private val engine: GuardrailsEngine
) {
    fun validate(toolCall: ToolCall): GuardrailResult {
        // 1. Check if tool is allowed
        if (toolCall.name !in allowedTools) {
            return GuardrailResult(
                action = GuardrailAction.BLOCK,
                riskScore = 1.0f,
                safeText = null,
                findings = emptyList(),
                messageToUser = "Tool '${toolCall.name}' is not allowed by policy."
            )
        }

        // 2. Scan arguments for injection/PII using the engine
        val result = engine.validateInput(toolCall.argumentsJson)
        
        // If sanitized, we might want to update arguments? 
        // For now, if action is SANITIZE, we return the sanitized args.
        // If BLOCK, we block the call.
        
        return result
    }
}
