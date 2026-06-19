package com.safellmkit.sdk

import com.safellmkit.core.policy.GuardrailsPolicies
import com.safellmkit.core.policy.GuardrailsPolicy

/**
 * SDK-level policy controls. Detection rules come from [guardrailsPolicy];
 * these flags control enforcement behavior at the provider gate.
 */
data class SdkPolicy(
    val guardrailsPolicy: GuardrailsPolicy = GuardrailsPolicies.strict(),
    /** When true, any guardrail/memory/provider error blocks the request (default). */
    val failClosedOnError: Boolean = true,
    /** When true, WARN on input prevents the provider call. */
    val blockOnWarn: Boolean = false
)
