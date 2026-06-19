package com.safellmkit.core.engine

import com.safellmkit.core.model.PromptFeatures
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

object PromptFeatureExtractor {

    private val jailbreakRegexes = listOf(
        Regex("(?i)\\b(dan|jailbreak|prompt injection|developer mode|roleplay|bypass safety)\\b"),
        Regex("(?i)(ignore|disregard|forget)\\s+(all\\s+)?(previous|prior)\\s+instructions"),
        Regex("(?i)reveal\\s+(your|the)\\s+(system prompt|instructions|policy)"),
        Regex("(?i)act as (?:a |an )?(developer|system|assistant)"),
        Regex("(?i)you are now"),
        Regex("(?i)do anything now")
    )

    private val piiRegexes = listOf(
        Regex("\\b[\\w.%+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b"),
        Regex("\\b(?:\\+?\\d{1,3}[\\s-]?)?(?:\\d{10}|\\d{3}[\\s-]?\\d{3}[\\s-]?\\d{4})\\b"),
        Regex("\\b(?:\\d[ -]*?){13,19}\\b")
    )

    fun extract(text: String): PromptFeatures {
        val normalized = text.trim()

        return PromptFeatures(
            jailbreakHits = countMatches(normalized, jailbreakRegexes),
            instructionOverrideHits = countMatches(
                normalized,
                listOf(
                    Regex("(?i)ignore previous instructions"),
                    Regex("(?i)disregard previous instructions"),
                    Regex("(?i)forget previous instructions"),
                    Regex("(?i)override safety"),
                    Regex("(?i)system prompt"),
                    Regex("(?i)developer mode")
                )
            ),
            piiHits = countMatches(normalized, piiRegexes),
            unicodeObfuscationScore = unicodeObfuscation(normalized),
            repetitionScore = repetitionScore(normalized),
            entropyScore = charEntropyScore(normalized),
            lengthScore = min(1f, normalized.length / 800f)
        )
    }

    fun heuristicRisk(features: PromptFeatures): Float {
        val keywordPressure = clamp01(features.jailbreakHits / 3f)
        val overridePressure = clamp01(features.instructionOverrideHits / 2f)
        val piiPressure = clamp01(features.piiHits / 2f)

        return clamp01(
            keywordPressure * 0.40f +
                overridePressure * 0.20f +
                piiPressure * 0.15f +
                features.unicodeObfuscationScore * 0.10f +
                features.repetitionScore * 0.08f +
                features.entropyScore * 0.05f +
                features.lengthScore * 0.02f
        )
    }

    private fun countMatches(text: String, regexes: List<Regex>): Int {
        var total = 0
        for (regex in regexes) {
            total += regex.findAll(text).count()
        }
        return total
    }

    private fun repetitionScore(text: String): Float {
        val tokens = text.lowercase()
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }

        if (tokens.size < 4) return 0f

        val distinctRatio = tokens.distinct().size.toFloat() / tokens.size.toFloat()
        return clamp01(1f - distinctRatio)
    }

    private fun unicodeObfuscation(text: String): Float {
        if (text.isEmpty()) return 0f

        var suspicious = 0
        for (ch in text) {
            val code = ch.code
            val zeroWidth = code == 0x200B || code == 0x200C || code == 0x200D || code == 0x2060
            val control = code < 32
            val weirdSymbol = !ch.isLetterOrDigit() &&
                !ch.isWhitespace() &&
                ch !in listOf('.', ',', ';', ':', '!', '?', '-', '_', '\'', '"', '(', ')', '[', ']', '{', '}')
            if (zeroWidth || control || weirdSymbol) suspicious++
        }
        return clamp01(suspicious.toFloat() / max(1, text.length).toFloat())
    }

    private fun charEntropyScore(text: String): Float {
        if (text.isBlank()) return 0f

        val frequencies = mutableMapOf<Char, Int>()
        for (ch in text.lowercase()) {
            frequencies[ch] = (frequencies[ch] ?: 0) + 1
        }

        val n = text.length.toFloat()
        var entropy = 0.0
        for (count in frequencies.values) {
            val p = count / n
            entropy -= p * ln(p.toDouble())
        }

        return clamp01((entropy / 4.5).toFloat())
    }

    private fun clamp01(v: Float): Float = max(0f, min(1f, v))
}