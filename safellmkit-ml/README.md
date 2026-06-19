# safellmkit-ml

ONNX runtime bindings, tokenizers, and **bundled model assets** for SafeLLMKit.

Consumed by `safellmkit-core` → `GuardrailEngine` → `OnnxRiskModel`, `TemporalRiskModel`, `MahalanobisDetector`.

---

## Targets

| Platform | ONNX inference | Notes |
|----------|----------------|-------|
| JVM | **Full** | `onnxruntime:1.18.0` |
| Android | **Full** | `onnxruntime-android:1.18.0`, minSdk 24 |
| iOS | **Stub / degraded** | `supportsInference = false` until native ORT integrated |

---

## Bundled assets

Path: `src/commonMain/resources/assets/`

| File | Purpose |
|------|---------|
| `guardrail_model_int8.onnx` | MiniLM 9-class guardrail classifier (INT8) |
| `temporal_model.onnx` | BiLSTM temporal risk (Crescendo / PAIR) |
| `gpt2_perplexity.onnx` | Perplexity scoring (optional path) |
| `label_map.json` | Class index → `AttackTaxonomy` name |
| `mean.npy`, `inv_cov.npy`, `class_centroids.npy` | Mahalanobis OOD statistics |
| `tokenizer/` | `vocab.txt`, `tokenizer.json`, `tokenizer_config.json` |

### 9-class taxonomy

| ID | Class |
|----|-------|
| 0 | SAFE |
| 1 | PROMPT_INJECTION |
| 2 | JAILBREAK |
| 3 | GCG |
| 4 | PAIR |
| 5 | CRESCENDO |
| 6 | ROLEPLAY |
| 7 | ENCODING_ATTACK |
| 8 | PII |

---

## Installation

```kotlin
dependencies {
    implementation("com.github.Aryan-Baglane.SafeLLMKit:safellmkit-ml:VERSION")
}
```

`safellmkit-core` already depends on this module.

---

## Runtime usage (internal)

Application code should **not** call these directly — they are wired inside `GuardrailEngine`:

```kotlin
// Internal defaults inside GuardrailEngine:
riskClassifier: RiskClassifier = OnnxRiskModel()
mahalanobisDetector: MahalanobisDetector = MahalanobisDetector()
temporalRiskModel: TemporalRiskModel = TemporalRiskModel()
```

### JVM

`OnnxRiskModel` loads `guardrail_model_int8.onnx` from classpath `/assets/`.

Requirements:

- Java 17+ may need JVM flag: `--enable-native-access=ALL-UNNAMED`
- Native ONNX Runtime library loaded automatically from Maven artifact

### Android

Add `safellmkit-ml` as a dependency. Assets are packaged in the AAR. Ensure ProGuard keeps ONNX and asset paths if minifying.

### iOS

Current `actual` implementations return safe defaults (`probability = 0`, `supportsInference = false`). Use server-side JVM inference or rules-only until ORT is integrated.

---

## Updating models after training

1. Train and export (see [ml-training/README.md](../ml-training/README.md))
2. Copy artifacts:

```bash
cp outputs/guardrail_model_int8.onnx safellmkit-ml/src/commonMain/resources/assets/
cp outputs/temporal_model.onnx safellmkit-ml/src/commonMain/resources/assets/
cp training_out/mahalanobis_stats.npz  # extract mean.npy, inv_cov.npy as needed
cp -r training_out/tokenizer/* safellmkit-ml/src/commonMain/resources/assets/tokenizer/
```

3. Rebuild: `./gradlew :safellmkit-ml:build`

---

## Tests

ML module has no dedicated test suite; inference is covered by `safellmkit-core` JVM tests (`GuardrailEngineTest.testRealModelInference`).

---

## Related

- [ml-training](../ml-training/README.md) — how models are trained
- [ML integration guide](../docs/ML_INTEGRATION.md)
- [safellmkit-core](../safellmkit-core/README.md)
