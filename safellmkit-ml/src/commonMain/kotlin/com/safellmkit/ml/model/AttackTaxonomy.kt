package com.safellmkit.ml.model

import kotlinx.serialization.Serializable

@Serializable
enum class AttackTaxonomy(val id: Int) {
    SAFE(0),
    PROMPT_INJECTION(1),
    JAILBREAK(2),
    GCG(3),
    PAIR(4),
    CRESCENDO(5),
    ROLEPLAY(6),
    ENCODING_ATTACK(7),
    PII(8),
    UNKNOWN(99);

    companion object {
        fun fromId(id: Int): AttackTaxonomy {
            return entries.firstOrNull { it.id == id } ?: UNKNOWN
        }
    }
}
