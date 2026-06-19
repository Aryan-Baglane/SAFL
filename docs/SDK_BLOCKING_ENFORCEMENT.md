# SafeLLMKit SDK — Blocking Enforcement Architecture

## Why blocking was not reliable

The core `GuardrailEngine` always computed correct `GuardrailAction` values, but **nothing in the library enforced them before LLM calls**. Integration code (and the sample web app) had to manually check `result.action` and skip the provider. In practice:

1. **Advisory-only API** — `GuardrailEngine.inspect()` and `GuardrailsAgent.protectInput()` returned results without gating provider calls.
2. **No provider layer in Kotlin** — OpenRouter was called directly from `sample-web-app` with separate heuristics; `SafeLLMKit` JS was instantiated but unused.
3. **Fail-open defaults** — `GuardrailEngine(policy = null)` disabled rule-based blocking; ML/Redis/iOS stubs could silently reduce risk signals.
4. **Alternate paths** — Any code calling `fetch(openrouter)` bypassed guardrails entirely.

Blocking failures were **architectural**: detection worked, enforcement did not exist as a single mandatory path.

## What changed

### Single public facade (`com.safellmkit.sdk`)

```kotlin
val client = SafeLLM.builder()
    .provider(OpenRouterProvider(apiKey, model))
    .build()

val response = client.chat("Hello")
```

`SafeLLMClient.chat()` is the only supported way to send prompts to a model.

### Central enforcement gate

`PolicyEnforcementGate.execute()` is the **only** code path that invokes `LlmProvider.generate()`:

```
User prompt
  → input inspect (GuardrailEngine)
  → if BLOCK → return ChatResponse(blocked=true), provider NOT called
  → if REDACT/SANITIZE → sanitize prompt
  → provider.generate(safe prompt)   ← only entry point
  → output inspect (GuardrailEngine)
  → if BLOCK → return blocked response, model text NOT returned
  → return ChatResponse(blocked=false, response=safe text)
```

### Fail-closed defaults

`SdkPolicy.failClosedOnError = true` (default):

- Guardrail inspection exception → `BLOCK`, provider not called
- Provider exception before/during call → `BLOCK`, no model text returned
- `GuardrailsPolicies.strict()` applied by default (rule-based injection blocking enabled)

### Provider wrappers

`OpenRouterProvider`, `OpenAIProvider`, `GeminiProvider`, and `OllamaProvider` perform HTTP only. They contain **no** guardrail logic and should not be called directly from application code.

## Where to look in code

| Concern | File |
|---------|------|
| Public entry | `sdk/SafeLLM.kt`, `sdk/SafeLLMClient.kt` |
| Enforcement gate | `sdk/PolicyEnforcementGate.kt` |
| Detection (unchanged) | `core/engine/GuardrailEngine.kt` |
| Provider HTTP | `sdk/provider/*` |
| Enforcement tests | `jvmTest/.../sdk/SafeLLMBlockingTest.kt` |

## Advanced / internal usage

`GuardrailEngine`, `GuardrailsAgent`, and memory types remain for custom pipelines and testing. SDK users should prefer `SafeLLMClient`.
