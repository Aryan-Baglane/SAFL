package com.safellmkit.core.agent

import com.safellmkit.core.GuardrailAction
import com.safellmkit.core.GuardrailFinding
import com.safellmkit.core.GuardrailResult
import com.safellmkit.core.GuardrailsEngine
import com.safellmkit.ml.NoOpSlmClassifier
import com.safellmkit.ml.SlmClassifier
import kotlin.math.max

@Deprecated(
    message = "Use com.safellmkit.sdk.SafeLLMClient for enforced chat. GuardrailsAgent does not gate provider calls.",
    replaceWith = ReplaceWith("SafeLLMClient", "com.safellmkit.sdk.SafeLLMClient")
)
class GuardrailsAgent(
    private val engine: GuardrailsEngine,
    private val config: GuardrailsAgentConfig = GuardrailsAgentConfig(),
    private val slm: SlmClassifier = NoOpSlmClassifier()
) {

    /**
     * Full protected call for INPUT stage (before LLM).
     */
    suspend fun protectInput(prompt: String): GuardrailResult {
        val base = engine.validateInput(prompt)

        // If already blocked, no need for ML
        if (base.action == GuardrailAction.BLOCK) return base

        // If ML is disabled
        if (!config.enableMlFallback) return base

        // If rules think it's totally safe, no need ML
        if (base.riskScore < config.rulesRiskThresholdToAskMl) return base

        // Ask ML for jailbreak probability
        val ml = slm.predict(prompt)
        val merged = merge(base, ml.label, ml.probability)

        return merged
    }

    /**
     * Full protected call for OUTPUT stage (after LLM).
     */
    suspend fun protectOutput(output: String): GuardrailResult {
        // Usually you don’t ML-check output; rules are enough.
        // But we keep it configurable.
        val base = engine.validateOutput(output)
        if (!config.enableMlFallback) return base

        if (base.action == GuardrailAction.BLOCK) return base
        if (base.riskScore < config.rulesRiskThresholdToAskMl) return base

        val ml = slm.predict(output)
        return merge(base, ml.label, ml.probability)
    }

    private fun merge(base: GuardrailResult, mlLabel: String, mlProb: Float): GuardrailResult {
        val mlFinding = GuardrailFinding(
            category = "ML_CLASSIFIER",
            rule = "SLM_JAILBREAK_CLASSIFIER",
            severity = when {
                mlProb >= config.mlRiskThresholdBlock -> 10
                mlProb >= config.mlRiskThresholdSanitize -> 7
                else -> 2
            },
            message = "ML verdict=$mlLabel, jailbreakProbability=$mlProb"
        )

        val newFindings = base.findings + mlFinding
        val boostedRisk = kotlin.math.max(base.riskScore, mlProb * 100f)

        val action = when {
            mlProb >= config.mlRiskThresholdBlock -> GuardrailAction.BLOCK
            mlProb >= config.mlRiskThresholdSanitize -> GuardrailAction.SANITIZE
            else -> base.action
        }

        val safeText = when (action) {
            GuardrailAction.ALLOW -> base.safeText
            GuardrailAction.SANITIZE -> base.safeText // rules already sanitized if needed
            GuardrailAction.BLOCK -> null
            GuardrailAction.WARN -> base.safeText
            GuardrailAction.REDACT -> base.safeText
        }

        val msg = when (action) {
            GuardrailAction.ALLOW -> base.messageToUser
            GuardrailAction.SANITIZE -> "Sanitized ⚠️ Suspicious prompt detected (ML confirmed)."
            GuardrailAction.BLOCK -> "Blocked 🚫 Jailbreak / policy bypass attempt detected (ML confirmed)."
            GuardrailAction.WARN -> "Warning ⚠️ Suspicious prompt detected (ML confirmed)."
            GuardrailAction.REDACT -> "Redacted ⚠️ Sensitive prompt content redacted (ML confirmed)."
        }

        return GuardrailResult(
            action = action,
            riskScore = boostedRisk,
            safeText = safeText,
            findings = newFindings,
            messageToUser = msg
        )
    }
}
