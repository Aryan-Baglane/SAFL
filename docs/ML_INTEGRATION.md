# ML Integration Guide

SafeLLMKit embeds ONNX models for on-device inference. The **primary classifier** is a MiniLM-based 9-class guardrail model; a **temporal BiLSTM** detects multi-turn Crescendo/PAIR patterns.

---

## Runtime stack

```
Prompt text
    → Tokenizer (BasicWordTokenizer / vocab from assets)
    → guardrail_model_int8.onnx  → logits + 384-d embedding
    → MahalanobisDetector (mean.npy, inv_cov.npy)
    → TemporalRiskModel (temporal_model.onnx, last 5 embeddings)
    → GuardrailEngine risk aggregation
```

| Asset | Location |
|-------|----------|
| `guardrail_model_int8.onnx` | `safellmkit-ml/.../assets/` |
| `temporal_model.onnx` | same |
| `mean.npy`, `inv_cov.npy` | Mahalanobis OOD |
| `label_map.json` | Class names |
| `tokenizer/*` | Inference tokenization |

---

## Kotlin wiring (automatic)

`GuardrailEngine` defaults:

```kotlin
riskClassifier: RiskClassifier = OnnxRiskModel()
mahalanobisDetector: MahalanobisDetector = MahalanobisDetector()
temporalRiskModel: TemporalRiskModel = TemporalRiskModel()
```

No manual setup required when using `SafeLLMClient` — models load from the classpath.

### Platform support

| Platform | MiniLM ONNX | Temporal | Mahalanobis |
|----------|-------------|----------|-------------|
| JVM | Yes | Yes | Yes |
| Android | Yes | Yes | Yes |
| iOS | Stub (degraded) | Stub | Returns 0 |

---

## 9-class taxonomy

See `label_map.json` and `AttackTaxonomy` enum:

SAFE, PROMPT_INJECTION, JAILBREAK, GCG, PAIR, CRESCENDO, ROLEPLAY, ENCODING_ATTACK, PII.

---

## SmoothLLM & perplexity

- **SmoothLLM:** `GuardrailEngine` perturbs the input (default 10 trials) and measures classifier variance when ONNX is available.
- **Perplexity:** character bigram heuristic by default; `gpt2_perplexity.onnx` available for future wiring.
- **GCG rule:** block when perplexity > 0.80 AND smooth variance > 0.50.

---

## Optional LLM validator

```kotlin
GuardrailEngine(
    llmValidatorFn = { text -> moderationApiScore(text) },
    policy = GuardrailsPolicies.strict()
)
```

Weight in aggregation: `llmWeight = 0.02f`. Math/ML layers dominate; validator is supplementary.

---

## Training & export

Full pipeline: [ml-training/README.md](../ml-training/README.md)

Quick reference:

```bash
cd ml-training
python train_minilm_guardrail.py --train data/train.jsonl --val data/val.jsonl --out outputs/run1
python export_onnx.py --model-dir outputs/run1 --out outputs/guardrail_model.onnx
python train_temporal_bilstm.py
# Copy ONNX + npy + tokenizer → safellmkit-ml/src/commonMain/resources/assets/
```

**Losses used in training:**

- Focal loss (γ=2) for class imbalance
- Batch-hard triplet loss on embeddings
- Mahalanobis stats computed post-training

---

## Legacy note

Older docs referenced `jailbreak_classifier.onnx` and `Md5Tokenizer`. The current runtime uses:

- `guardrail_model_int8.onnx`
- `BasicWordTokenizer` + HuggingFace-style `vocab.txt`
- `OnnxRiskModel` (not `OnnxJvmClassifier` / `GuardrailsAgent`)

Python package may still accept a custom ONNX path via `OnnxJailbreakClassifier`.

---

## Verify inference

```bash
./gradlew :safellmkit-core:jvmTest --tests "com.safellmkit.core.engine.GuardrailEngineTest.testRealModelInference"
```

---

## Related

- [safellmkit-ml/README.md](../safellmkit-ml/README.md)
- [Root README](../README.md)
