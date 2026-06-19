package com.safellmkit.core.engine

import com.safellmkit.core.model.*
import com.safellmkit.core.policy.*
import com.safellmkit.ml.onnx.OnnxRiskModel
import com.safellmkit.ml.ood.MahalanobisDetector
import com.safellmkit.ml.model.RiskClassifier
import com.safellmkit.ml.temporal.TemporalRiskModel
import kotlin.math.sqrt

interface EmbeddingProvider {
    fun embed(text: String, dimensions: Int = 384): FloatArray
}

/**
 * Internal detection engine. SDK users should use [com.safellmkit.sdk.SafeLLMClient] which
 * enforces [inspect] results before any provider call.
 */
@OptIn(kotlin.ExperimentalStdlibApi::class)
class GuardrailEngine(
    val config: GuardrailConfig = GuardrailConfig(),
    val memory: ConversationMemoryGateway = InMemoryConversationMemory(),
    val userMemory: com.safellmkit.core.memory.UserMemoryEngine = com.safellmkit.core.memory.InMemoryUserMemoryEngine(),
    val embeddingProvider: EmbeddingProvider? = null,
    val perplexityScoreFn: ((String) -> Float)? = null,
    val smoothVarianceScoreFn: ((String) -> Float)? = null,
    val mlProbabilityFn: ((String) -> Float?)? = null,
    val mahalanobisScoreFn: ((FloatArray) -> Float)? = null,
    val temporalRiskFn: ((List<ConversationTurn>) -> Float)? = null,
    val llmValidatorFn: (suspend (String) -> Float?)? = null,
    val nearestAttackSimilarityFn: ((FloatArray) -> Float)? = null,
    val policy: GuardrailsPolicy? = null,
    val riskClassifier: RiskClassifier = OnnxRiskModel(),
    val mahalanobisDetector: MahalanobisDetector = MahalanobisDetector(),
    val temporalRiskModel: TemporalRiskModel = TemporalRiskModel()
) : AutoCloseable {

    init {
        val diag = loadCentroidsDiagnostics()
        println("[GuardrailEngine] $diag")
    }

    private fun perturbString(text: String, perturbationPct: Float = 0.10f): String {
        if (text.isEmpty()) return text
        val chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        val textList = text.toCharArray()
        val numToSwap = maxOf(1, (textList.size * perturbationPct).toInt())
        val shuffledIndices = (0 until textList.size).toList().shuffled().take(minOf(numToSwap, textList.size))
        for (idx in shuffledIndices) {
            val char = textList[idx]
            val isAlnum = char in 'a'..'z' || char in 'A'..'Z' || char in '0'..'9'
            if (isAlnum) {
                textList[idx] = chars.random()
            }
        }
        return textList.concatToString()
    }

    private fun estimateCharacterPerplexity(text: String): Float {
        if (text.length < 2) return 0f
        val counts = mutableMapOf<String, Int>()
        var total = 0
        for (i in 0 until text.length - 1) {
            val bigram = text.substring(i, i + 2)
            counts[bigram] = (counts[bigram] ?: 0) + 1
            total++
        }
        var entropy = 0.0
        for (count in counts.values) {
            val p = count.toDouble() / total
            entropy -= p * (kotlin.math.log2(p))
        }
        val lengthFactor = (text.length / 150f).coerceIn(0f, 1f)
        val score = (((entropy - 5.2) / 1.5).toFloat() * lengthFactor).coerceIn(0f, 1f)
        return score
    }

    @Deprecated("Use inspect instead", ReplaceWith("inspect(turn)"))
    suspend fun validateInput(prompt: String): GuardrailResult {
        return inspect(ConversationTurn(
            sessionId = "default-session",
            turnIndex = 0,
            content = prompt
        ))
    }

    @Deprecated("Use inspect instead", ReplaceWith("inspect(turn)"))
    suspend fun validateInput(turn: ConversationTurn): GuardrailResult {
        return inspect(turn)
    }

    suspend fun inspect(turn: ConversationTurn): GuardrailResult {
        // 1. FeatureExtractor & RuleEngine
        val features = PromptFeatureExtractor.extract(turn.content)
        
        val activeRules = policy?.let { p ->
            if (turn.role == "assistant") p.outputRules else p.inputRules
        } ?: emptyList()

        val findings = activeRules.flatMap { rp ->
            rp.rule.check(turn.content).filter { it.severity >= rp.minSeverityToTrigger }
        }

        val heuristicRisk = PromptFeatureExtractor.heuristicRisk(features)

        // 2. MiniLMEngine
        val prediction = if (riskClassifier.supportsInference) {
            riskClassifier.predict(turn.content)
        } else {
            null
        }

        val mlProbability = prediction?.probability ?: mlProbabilityFn?.invoke(turn.content)?.coerceIn(0f, 1f)
        val mlRisk = mlProbability ?: 0f

        var attackClass = prediction?.attackClass ?: when {
            features.jailbreakHits > 0 -> AttackTaxonomy.JAILBREAK
            features.instructionOverrideHits > 0 -> AttackTaxonomy.PROMPT_INJECTION
            mlProbability != null && mlProbability > 0.8f -> AttackTaxonomy.JAILBREAK
            else -> AttackTaxonomy.SAFE
        }

        // 3. Embedding Provider
        val currentEmbedding = if (embeddingProvider != null) {
            embeddingProvider.embed(turn.content)
        } else if (prediction != null && prediction.embedding.isNotEmpty()) {
            prediction.embedding
        } else {
            HashingEmbeddingProvider().embed(turn.content)
        }

        // User Memory Gateway / Cooldown Setup (Phase 3)
        val userId = turn.userId
        userMemory.appendSession(userId, turn.sessionId)
        val rawUserState = userMemory.getUserState(userId)
        val currentTimestamp = if (turn.timestampMs > 0L) turn.timestampMs else currentTimeMs()
        
        var decayedReputation = rawUserState.reputationScore
        if (rawUserState.lastUpdated > 0L && currentTimestamp > rawUserState.lastUpdated) {
            val seconds = (currentTimestamp - rawUserState.lastUpdated) / 1000f
            decayedReputation *= kotlin.math.exp(-seconds / 30f)
        }

        val hasUserCooledDown = (rawUserState.lastUpdated > 0L && (currentTimestamp - rawUserState.lastUpdated) / 1000f >= 30f)
        val previousUserState = if (hasUserCooledDown) {
            rawUserState.copy(
                attackCount = 0,
                blockedCount = 0,
                conversationStatus = ConversationStatus.NORMAL,
                reputationScore = decayedReputation
            )
        } else {
            rawUserState.copy(
                reputationScore = decayedReputation
            )
        }
        val isUserBlockedSessionStart = previousUserState.conversationStatus == ConversationStatus.BLOCKED

        // 4. MemoryEngine
        val centroid = if (currentEmbedding.isNotEmpty()) {
            memory.rollingCentroid(turn.sessionId, config.historyWindow)
        } else null
        
        val semanticSimilarity = centroid?.let { cosine(it, currentEmbedding).coerceIn(0f, 1f) } ?: 1f
        val semanticDriftScore = 1f - semanticSimilarity
        val driftRisk = semanticDriftScore

        // Cross-Session Embedding Drift (Phase 4)
        val userCentroid = if (previousUserState.embeddingHistory.isNotEmpty()) {
            val dim = previousUserState.embeddingHistory[0].size
            val c = FloatArray(dim)
            for (emb in previousUserState.embeddingHistory) {
                for (i in 0 until dim) {
                    c[i] += emb[i]
                }
            }
            for (i in 0 until dim) {
                c[i] /= previousUserState.embeddingHistory.size.toFloat()
            }
            c
        } else null

        val crossSessionSimilarity = if (userCentroid != null && currentEmbedding.isNotEmpty()) {
            cosine(userCentroid, currentEmbedding).coerceIn(0f, 1f)
        } else 1f
        val crossSessionDrift = 1f - crossSessionSimilarity

        // 5. PerplexityEngine
        val perplexityScore = perplexityScoreFn?.invoke(turn.content)?.coerceIn(0f, 1f)
            ?: estimateCharacterPerplexity(turn.content)
        val perplexityRisk = perplexityScore

        // 6. SmoothLLMEngine
        val (smoothVariance, smoothMean) = if (smoothVarianceScoreFn != null) {
            val score = smoothVarianceScoreFn.invoke(turn.content).coerceIn(0f, 1f)
            Pair(score, score)
        } else {
            if (riskClassifier.supportsInference) {
                val numPerturbations = 10
                val probs = ArrayList<Float>(numPerturbations)
                for (i in 0 until numPerturbations) {
                    val perturbed = perturbString(turn.content)
                    val pred = riskClassifier.predict(perturbed)
                    probs.add(pred.probability)
                }
                val mean = probs.average().toFloat()
                val variance = if (probs.size > 1) {
                    val m = probs.average()
                    probs.map { (it - m) * (it - m) }.sum().toFloat() / probs.size
                } else 0f
                Pair(variance, mean)
            } else {
                Pair(0f, 0f)
            }
        }
        val smoothVarianceScore = smoothVariance
        val smoothRisk = smoothMean

        // 7. MahalanobisEngine
        val mahalanobisScore = if (currentEmbedding.isNotEmpty()) {
            mahalanobisScoreFn?.invoke(currentEmbedding)?.coerceAtLeast(0f)
                ?: mahalanobisDetector.score(currentEmbedding)
        } else 0f

        val dim = currentEmbedding.size
        val mahalanobisRisk = when {
            mahalanobisScore <= 0f -> 0f
            dim <= 16 -> if (mahalanobisScore > 6f) 1f else (mahalanobisScore / 6f).coerceIn(0f, 1f)
            else -> {
                val expectedMean = sqrt(dim.toDouble()).toFloat()
                val lowerThreshold = expectedMean * 1.6f
                val upperThreshold = expectedMean * 1.9f
                ((mahalanobisScore - lowerThreshold) / (upperThreshold - lowerThreshold)).coerceIn(0f, 1f)
            }
        }

        // 8. VectorSearchEngine
        val nearestAttackSimilarity = if (currentEmbedding.isNotEmpty()) {
            nearestAttackSimilarityFn?.invoke(currentEmbedding)?.coerceIn(0f, 1f)
                ?: memory.queryVectorSimilarity(currentEmbedding).coerceIn(0f, 1f)
        } else 0f
        val vectorSearchRisk = nearestAttackSimilarity

        // Cross-Session Vector Search (Phase 5)
        val nearestUserAttackSimilarity = if (currentEmbedding.isNotEmpty()) {
            userMemory.queryVectorSimilarity(userId, currentEmbedding).coerceIn(0f, 1f)
        } else 0f

        // 9. TemporalEngine
        val recentEmbeddings = memory.recentEmbeddings(turn.sessionId, 5)
        val sequenceForTemporal = (recentEmbeddings + currentEmbedding).takeLast(5)
        val temporalRiskPrediction = if (temporalRiskModel.supportsInference && sequenceForTemporal.isNotEmpty()) {
            temporalRiskModel.predict(sequenceForTemporal)
        } else {
            null
        }
        val recentTurnsList = memory.recentTurns(turn.sessionId, config.historyWindow)
        val temporalRisk = temporalRiskFn?.invoke(recentTurnsList)?.coerceIn(0f, 1f)
            ?: temporalRiskPrediction?.let { maxOf(it.crescendoProbability, it.pairProbability) }
            ?: 0f

        if (temporalRiskPrediction != null) {
            if (temporalRiskPrediction.crescendoProbability > 0.8f) {
                attackClass = AttackTaxonomy.CRESCENDO
            } else if (temporalRiskPrediction.pairProbability > 0.8f) {
                attackClass = AttackTaxonomy.PAIR
            }
        }

        if (perplexityScore > 0.80f && smoothVarianceScore > 0.50f) {
            attackClass = AttackTaxonomy.GCG
        }

        // 10. LLMValidator
        val llmRisk = llmValidatorFn?.invoke(turn.content)?.coerceIn(0f, 1f)

        // 11. RiskAggregator
        val totalWeightUsed = if (llmRisk != null) 1.0f else (1.0f - config.llmWeight)
        val rawRisk = (
            heuristicRisk * config.heuristicWeight +
            driftRisk * config.driftWeight +
            perplexityRisk * config.perplexityWeight +
            smoothRisk * config.smoothWeight +
            mahalanobisRisk * config.mahalanobisWeight +
            mlRisk * config.mlWeight +
            temporalRisk * config.temporalWeight +
            vectorSearchRisk * config.vectorWeight +
            (llmRisk ?: 0f) * config.llmWeight
        )
        val combinedRisk = (rawRisk / totalWeightUsed).coerceIn(0f, 1f)

        // Reputation score update: R_new = 0.9 * R_old + 0.1 * attackScore (Phase 3)
        val userReputationScore = (0.9f * decayedReputation + 0.1f * combinedRisk).coerceIn(0f, 1f)

        // Cross-session risk blend
        val crossSessionRisk = (userReputationScore * 0.5f + crossSessionDrift * 0.2f + nearestUserAttackSimilarity * 0.3f).coerceIn(0f, 1f)

        // Phase 8: Final blended risk formula
        val blendedRisk = (0.95f * combinedRisk + 0.05f * crossSessionRisk).coerceIn(0f, 1f)

        // Stateful mitigation and exponential decay scaling
        val previousState = memory.getConversationState(turn.sessionId)
        val previousStatus = previousState?.status ?: ConversationStatus.NORMAL
        
        val rOld = previousState?.lastRiskScore ?: 0f
        val rNew = 0.9f * rOld + 0.1f * blendedRisk

        val statusScaleFactor = when (previousStatus) {
            ConversationStatus.BLOCKED -> 1.5f
            ConversationStatus.ACTIVE_ATTACK -> 1.25f
            ConversationStatus.SUSPICIOUS -> 1.1f
            else -> 1.0f
        }
        val scaledRisk = (blendedRisk * statusScaleFactor).coerceIn(0f, 1f)

        // Determine Action
        val ruleAction = if (findings.isEmpty()) GuardrailAction.ALLOW else {
            val blocked = activeRules.any { rp ->
                rp.mode == com.safellmkit.core.policy.GuardrailMode.BLOCK_IF_FOUND &&
                        findings.any { it.rule == rp.rule.name() }
            }
            if (blocked) GuardrailAction.BLOCK else {
                val sanitize = activeRules.any { rp ->
                    rp.mode == com.safellmkit.core.policy.GuardrailMode.SANITIZE_IF_FOUND &&
                            findings.any { it.rule == rp.rule.name() }
                }
                if (sanitize) GuardrailAction.SANITIZE else GuardrailAction.ALLOW
            }
        }

        val mathAction = when {
            previousStatus == ConversationStatus.BLOCKED -> GuardrailAction.BLOCK
            isUserBlockedSessionStart -> GuardrailAction.BLOCK
            attackClass == AttackTaxonomy.GCG -> GuardrailAction.BLOCK
            attackClass != AttackTaxonomy.SAFE && mlProbability != null && mlProbability >= config.blockThreshold -> GuardrailAction.BLOCK
            scaledRisk >= config.blockThreshold -> GuardrailAction.BLOCK
            features.piiHits > 0 && scaledRisk >= config.warnThreshold -> GuardrailAction.REDACT
            attackClass != AttackTaxonomy.SAFE && mlProbability != null && mlProbability >= config.warnThreshold -> GuardrailAction.WARN
            scaledRisk >= config.warnThreshold -> GuardrailAction.WARN
            else -> GuardrailAction.ALLOW
        }

        val action = when {
            previousStatus == ConversationStatus.BLOCKED -> GuardrailAction.BLOCK
            isUserBlockedSessionStart -> GuardrailAction.BLOCK
            ruleAction == GuardrailAction.BLOCK || mathAction == GuardrailAction.BLOCK -> GuardrailAction.BLOCK
            ruleAction == GuardrailAction.SANITIZE -> GuardrailAction.SANITIZE
            mathAction == GuardrailAction.REDACT -> GuardrailAction.REDACT
            mathAction == GuardrailAction.WARN -> GuardrailAction.WARN
            else -> GuardrailAction.ALLOW
        }

        val safeText = when (action) {
            GuardrailAction.ALLOW -> turn.content
            GuardrailAction.REDACT -> sanitizeAll(turn.content, activeRules)
            GuardrailAction.BLOCK -> null
            GuardrailAction.WARN -> turn.content
            GuardrailAction.SANITIZE -> sanitizeAll(turn.content, activeRules)
        }

        // Append to memory
        if (currentEmbedding.isNotEmpty()) {
            memory.append(turn, currentEmbedding)
        }

        // Update user state machine status (Phase 7)
        val isUserBlocked = (
            isUserBlockedSessionStart ||
            action == GuardrailAction.BLOCK ||
            userReputationScore >= config.blockThreshold
        )
        val newUserStatus = when {
            isUserBlocked -> ConversationStatus.BLOCKED
            userReputationScore >= config.warnThreshold || (previousUserState.attackCount + (if (attackClass != AttackTaxonomy.SAFE) 1 else 0)) >= 3 -> ConversationStatus.ACTIVE_ATTACK
            userReputationScore >= 0.35f || (previousUserState.attackCount + (if (attackClass != AttackTaxonomy.SAFE) 1 else 0)) >= 1 -> ConversationStatus.SUSPICIOUS
            userReputationScore > 0.10f -> ConversationStatus.COOLDOWN
            else -> ConversationStatus.NORMAL
        }

        val newUserState = previousUserState.copy(
            attackCount = previousUserState.attackCount + (if (attackClass != AttackTaxonomy.SAFE) 1 else 0),
            blockedCount = previousUserState.blockedCount + (if (action == GuardrailAction.BLOCK) 1 else 0),
            riskHistory = previousUserState.riskHistory + blendedRisk,
            taxonomyHistory = previousUserState.taxonomyHistory + attackClass,
            reputationScore = userReputationScore,
            lastUpdated = currentTimestamp,
            conversationStatus = newUserStatus
        )
        userMemory.updateUserState(newUserState)
        userMemory.appendEmbedding(userId, turn.sessionId, currentEmbedding, attackClass, currentTimestamp)

        // Update conversation state counters
        val newAttackCount = (previousState?.attackCount ?: 0) + (if (attackClass != AttackTaxonomy.SAFE) 1 else 0)
        val newBlockedCount = (previousState?.blockedCount ?: 0) + (if (action == GuardrailAction.BLOCK) 1 else 0)

        val newStatus = when {
            action == GuardrailAction.BLOCK || rNew >= config.blockThreshold || previousStatus == ConversationStatus.BLOCKED -> ConversationStatus.BLOCKED
            rNew >= config.warnThreshold -> ConversationStatus.ACTIVE_ATTACK
            rNew >= 0.35f -> ConversationStatus.SUSPICIOUS
            else -> ConversationStatus.NORMAL
        }

        memory.updateConversationState(
            ConversationState(
                sessionId = turn.sessionId,
                totalTurns = (previousState?.totalTurns ?: 0) + 1,
                lastTurnIndex = turn.turnIndex,
                lastRiskScore = rNew,
                lastSemanticSimilarity = semanticSimilarity,
                lastPerplexityScore = perplexityScore,
                lastSmoothVarianceScore = smoothVarianceScore,
                lastMahalanobisScore = mahalanobisScore,
                lastAction = action,
                status = newStatus,
                lastAttackClass = attackClass,
                attackCount = newAttackCount,
                blockedCount = newBlockedCount
            )
        )

        // Reasons & Explainability (Phase 9)
        val reasons = buildList {
            if (features.jailbreakHits > 0) add("jailbreak keywords detected")
            if (features.instructionOverrideHits > 0) add("instruction override detected")
            if (features.piiHits > 0) add("possible PII detected")
            if (semanticSimilarity < config.driftThreshold) add("semantic drift across turns")
            if (perplexityScore > 0.65f) add("prompt anomaly / perplexity spike")
            if (smoothVarianceScore > 0.30f) add("SmoothLLM instability detected")
            if (mahalanobisScore > 6f) add("latent-space anomaly detected")
            if (mlProbability != null && mlProbability > 0.80f) add("ML classifier high confidence")
            if (temporalRisk > 0.80f) add("temporal multi-turn attack pattern detected")
            if (nearestAttackSimilarity > 0.80f) add("vector similarity to known attacks")
            
            // Cross-session explains
            if (previousUserState.attackCount > 0) {
                add("User previously triggered ${previousUserState.attackCount} attacks")
            }
            if (nearestUserAttackSimilarity > 0.80f) {
                add("Similarity to prior attack cluster = $nearestUserAttackSimilarity")
            }
            if (crossSessionDrift > 0.40f) {
                add("Cross-session drift detected")
            }
        }.ifEmpty { listOf("no strong attack indicators") }

        val baseHeuristicRisk = findings.sumOf { rp -> rp.severity * 10 }.toFloat()
        val mathRiskVal = if (scaledRisk < 0.20f) scaledRisk * 0.4f else scaledRisk
        val mathRiskScore = mathRiskVal * 100f
        val finalRiskScore = maxOf(baseHeuristicRisk, mathRiskScore).coerceIn(0f, 100f)

        return GuardrailResult(
            action = action,
            riskScore = finalRiskScore,
            safeText = safeText,
            findings = findings,
            messageToUser = reasons.joinToString(", "),
            attackClass = attackClass,
            attackProbability = mlProbability ?: 0f,
            embedding = if (currentEmbedding.isNotEmpty()) currentEmbedding else null,
            heuristicRisk = heuristicRisk,
            driftRisk = driftRisk,
            perplexityRisk = perplexityRisk,
            smoothRisk = smoothRisk,
            mahalanobisRisk = mahalanobisRisk,
            mlRisk = mlRisk,
            temporalRisk = temporalRisk,
            llmRisk = llmRisk,
            semanticSimilarity = semanticSimilarity,
            semanticDriftScore = semanticDriftScore,
            perplexityScore = perplexityScore,
            smoothVarianceScore = smoothVarianceScore,
            mahalanobisScore = mahalanobisScore,
            mlProbability = mlProbability,
            nearestAttackSimilarity = nearestAttackSimilarity,
            reasons = reasons,
            featureSnapshot = features,

            // Phase 9 new outputs
            userReputationScore = userReputationScore,
            crossSessionDrift = crossSessionDrift,
            crossSessionRisk = crossSessionRisk,
            userState = newUserState
        )
    }

    private fun sanitizeAll(text: String, rules: List<RulePolicy>): String {
        var safe = text
        rules.forEach { rp ->
            safe = rp.rule.sanitize(safe)
        }
        return safe
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        val n = minOf(a.size, b.size)
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in 0 until n) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        if (na == 0f || nb == 0f) return 0f
        return (dot / (sqrt(na) * sqrt(nb))).coerceIn(-1f, 1f)
    }

    override fun close() {
        if (riskClassifier is AutoCloseable) {
            riskClassifier.close()
        }
        temporalRiskModel.close()
    }
}
