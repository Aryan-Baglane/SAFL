package com.safellmkit.core.model

import com.safellmkit.core.compliance.OwaspLlmTop10
import kotlinx.serialization.Serializable

@Serializable
enum class GuardrailAction {
    ALLOW,
    WARN,
    BLOCK,
    REDACT,
    @Deprecated("Use REDACT or WARN instead")
    SANITIZE
}

typealias AttackTaxonomy = com.safellmkit.ml.model.AttackTaxonomy

@Serializable
enum class ConversationStatus {
    NORMAL,
    SUSPICIOUS,
    ACTIVE_ATTACK,
    BLOCKED,
    COOLDOWN
}

@Serializable
data class ConversationTurn(
    val sessionId: String,
    val turnIndex: Int,
    val role: String = "user",
    val content: String,
    val additionalContext: List<String> = emptyList(),
    val timestampMs: Long = 0L,
    val userId: String = "default-user"
)

@Serializable
data class UserState(
    val userId: String,
    val activeSessions: List<String> = emptyList(),
    val attackCount: Int = 0,
    val blockedCount: Int = 0,
    val riskHistory: List<Float> = emptyList(),
    val taxonomyHistory: List<AttackTaxonomy> = emptyList(),
    val sessionHistory: List<String> = emptyList(),
    val reputationScore: Float = 0f,
    val lastUpdated: Long = 0L,
    val lastAttackTimestampMs: Long = 0L,
    val conversationStatus: ConversationStatus = ConversationStatus.NORMAL,
    val embeddingHistory: List<List<Float>> = emptyList(),
    val recentAttackVectors: List<List<Float>> = emptyList()
)

@Serializable
data class PromptFeatures(
    val jailbreakHits: Int,
    val instructionOverrideHits: Int,
    val piiHits: Int,
    val unicodeObfuscationScore: Float,
    val repetitionScore: Float,
    val entropyScore: Float,
    val lengthScore: Float
)

@Serializable
data class GuardrailConfig(
    val historyWindow: Int = 8,
    val warnThreshold: Float = 0.55f,
    val blockThreshold: Float = 0.80f,
    val driftThreshold: Float = 0.65f,
    val heuristicWeight: Float = 0.15f,
    val driftWeight: Float = 0.10f,
    val perplexityWeight: Float = 0.15f,
    val smoothWeight: Float = 0.10f,
    val mahalanobisWeight: Float = 0.10f,
    val mlWeight: Float = 0.20f,
    val temporalWeight: Float = 0.15f,
    val vectorWeight: Float = 0.03f,
    val llmWeight: Float = 0.02f,
    val classifierBlockThreshold: Float = 0.85f,
    val temporalBlockThreshold: Float = 0.80f,
    val mahalanobisBlockThreshold: Float = 40.0f,
    val reputationBlockThreshold: Float = 0.85f,
    val reputationDecayInactivityMs: Long = 6L * 60L * 60L * 1000L,
    val reputationDecayFactorOnInactivity: Float = 0.5f,
    val stepDriftWeight: Float = 0.05f,
    val cumulativeDriftWeight: Float = 0.05f,
    val entropyMarginThreshold: Float = 0.20f,
    val uncertaintyPenalty: Float = 0.40f
)

@Serializable
data class GuardrailFinding(
    val category: String,
    val rule: String,
    val severity: Int, // 1..10
    val message: String,
    val owaspMapping: OwaspLlmTop10? = null
)

@Serializable
data class GuardrailResult(
    val action: GuardrailAction,
    val riskScore: Float,
    val safeText: String? = null,
    val findings: List<GuardrailFinding> = emptyList(),
    val messageToUser: String = "",
    val attackClass: AttackTaxonomy = AttackTaxonomy.SAFE,
    val attackProbability: Float = 0f,
    val embedding: FloatArray? = null,
    
    // Risk Breakdown Contributors for Explainability
    val heuristicRisk: Float = 0f,
    val driftRisk: Float = 0f,
    val perplexityRisk: Float = 0f,
    val smoothRisk: Float = 0f,
    val mahalanobisRisk: Float = 0f,
    val mlRisk: Float = 0f,
    val temporalRisk: Float = 0f,
    val llmRisk: Float? = null,
    
    // Mathematical Scores
    val semanticSimilarity: Float = 1f,
    val semanticDriftScore: Float = 0f,
    val perplexityScore: Float = 0f,
    val smoothVarianceScore: Float = 0f,
    val mahalanobisScore: Float = 0f,
    val mlProbability: Float? = null,
    val nearestAttackSimilarity: Float = 0f,
    val reasons: List<String> = emptyList(),
    val featureSnapshot: PromptFeatures? = null,

    // Cross-Session Explainability fields
    val userReputationScore: Float = 0f,
    val crossSessionDrift: Float = 0f,
    val crossSessionRisk: Float = 0f,
    val userState: UserState? = null,
    val stepDrift: Float = 0f,
    val cumulativeDrift: Float = 0f
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as GuardrailResult

        if (action != other.action) return false
        if (riskScore != other.riskScore) return false
        if (safeText != other.safeText) return false
        if (findings != other.findings) return false
        if (messageToUser != other.messageToUser) return false
        if (attackClass != other.attackClass) return false
        if (attackProbability != other.attackProbability) return false
        if (embedding != null) {
            if (other.embedding == null) return false
            if (!embedding.contentEquals(other.embedding)) return false
        } else if (other.embedding != null) return false
        if (heuristicRisk != other.heuristicRisk) return false
        if (driftRisk != other.driftRisk) return false
        if (perplexityRisk != other.perplexityRisk) return false
        if (smoothRisk != other.smoothRisk) return false
        if (mahalanobisRisk != other.mahalanobisRisk) return false
        if (mlRisk != other.mlRisk) return false
        if (temporalRisk != other.temporalRisk) return false
        if (llmRisk != other.llmRisk) return false
        if (semanticSimilarity != other.semanticSimilarity) return false
        if (semanticDriftScore != other.semanticDriftScore) return false
        if (perplexityScore != other.perplexityScore) return false
        if (smoothVarianceScore != other.smoothVarianceScore) return false
        if (mahalanobisScore != other.mahalanobisScore) return false
        if (mlProbability != other.mlProbability) return false
        if (nearestAttackSimilarity != other.nearestAttackSimilarity) return false
        if (reasons != other.reasons) return false
        if (featureSnapshot != other.featureSnapshot) return false
        if (userReputationScore != other.userReputationScore) return false
        if (crossSessionDrift != other.crossSessionDrift) return false
        if (crossSessionRisk != other.crossSessionRisk) return false
        if (userState != other.userState) return false
        if (stepDrift != other.stepDrift) return false
        if (cumulativeDrift != other.cumulativeDrift) return false

        return true
    }

    override fun hashCode(): Int {
        var result = action.hashCode()
        result = 31 * result + riskScore.hashCode()
        result = 31 * result + (safeText?.hashCode() ?: 0)
        result = 31 * result + findings.hashCode()
        result = 31 * result + messageToUser.hashCode()
        result = 31 * result + attackClass.hashCode()
        result = 31 * result + attackProbability.hashCode()
        result = 31 * result + (embedding?.contentHashCode() ?: 0)
        result = 31 * result + heuristicRisk.hashCode()
        result = 31 * result + driftRisk.hashCode()
        result = 31 * result + perplexityRisk.hashCode()
        result = 31 * result + smoothRisk.hashCode()
        result = 31 * result + mahalanobisRisk.hashCode()
        result = 31 * result + mlRisk.hashCode()
        result = 31 * result + temporalRisk.hashCode()
        result = 31 * result + (llmRisk?.hashCode() ?: 0)
        result = 31 * result + semanticSimilarity.hashCode()
        result = 31 * result + semanticDriftScore.hashCode()
        result = 31 * result + perplexityScore.hashCode()
        result = 31 * result + smoothVarianceScore.hashCode()
        result = 31 * result + mahalanobisScore.hashCode()
        result = 31 * result + (mlProbability?.hashCode() ?: 0)
        result = 31 * result + nearestAttackSimilarity.hashCode()
        result = 31 * result + reasons.hashCode()
        result = 31 * result + (featureSnapshot?.hashCode() ?: 0)
        result = 31 * result + userReputationScore.hashCode()
        result = 31 * result + crossSessionDrift.hashCode()
        result = 31 * result + crossSessionRisk.hashCode()
        result = 31 * result + (userState?.hashCode() ?: 0)
        result = 31 * result + stepDrift.hashCode()
        result = 31 * result + cumulativeDrift.hashCode()
        return result
    }

    fun toExplainabilityReport(): String {
        return kotlinx.serialization.json.Json { prettyPrint = true }.encodeToString(serializer(), this)
    }
}

@Serializable
data class ConversationState(
    val sessionId: String,
    val totalTurns: Int,
    val lastTurnIndex: Int,
    val lastRiskScore: Float,
    val lastSemanticSimilarity: Float,
    val lastPerplexityScore: Float = 0f,
    val lastSmoothVarianceScore: Float = 0f,
    val lastMahalanobisScore: Float = 0f,
    val lastAction: GuardrailAction,
    val status: ConversationStatus,
    
    // Crescendo / Multi-turn attack memory counters
    val lastAttackClass: AttackTaxonomy = AttackTaxonomy.SAFE,
    val attackCount: Int = 0,
    val blockedCount: Int = 0
)

@Serializable
data class ConversationSnapshot(
    val sessionId: String,
    val turns: List<ConversationTurn>,
    val state: ConversationState
)
