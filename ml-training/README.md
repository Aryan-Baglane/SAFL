# ml-training

**Training-only** scripts for SafeLLMKit models. Nothing in this folder ships to production apps.

Outputs are copied into `safellmkit-ml/src/commonMain/resources/assets/`.

---

## Prerequisites

```bash
cd ml-training
python -m venv .venv
source .venv/bin/activate
pip install torch transformers onnx numpy
```

GPU optional (CUDA/MPS); scripts fall back to CPU.

---

## Dataset layout

Use `build_dataset.py` to normalize raw data into JSONL/CSV.

### 9-class labels

| Label | Name |
|-------|------|
| 0 | safe |
| 1 | prompt_injection |
| 2 | jailbreak |
| 3 | gcg |
| 4 | crescendo |
| 5 | pair |
| 6 | roleplay |
| 7 | encoding_attack |
| 8 | pii |

### Build a unified dataset

```bash
python build_dataset.py \
  --input ./raw_data \
  --out ./data/guardrail_train.jsonl \
  --format jsonl
```

Expected row fields: `text` (or `prompt`/`content`), `label`, optional `session_id`, `turn_index`.

**Dataset sources:** organize folders by attack type or supply explicit `label` columns. The builder infers labels from folder names using `LABEL_DEFAULTS` in `build_dataset.py`.

---

## Train MiniLM 9-class guardrail

Script: `train_minilm_guardrail.py`

Architecture: `sentence-transformers/all-MiniLM-L6-v2` encoder + projection head + 9-way classifier.

**Losses:**

- **Focal loss** (`focal_loss.py`, γ=2) — handles class imbalance
- **Triplet loss** (batch-hard, margin=0.3, weight=0.2) — separates embeddings by attack class

**Mahalanobis stats** computed after training → `mahalanobis_stats.npz` (`mean`, `inv_cov`).

```bash
python train_minilm_guardrail.py \
  --train ./data/train.jsonl \
  --val ./data/val.jsonl \
  --out ./outputs/minilm_run \
  --num-labels 9 \
  --epochs 4 \
  --batch 32
```

Outputs in `--out`:

| File | Use |
|------|-----|
| `minilm_guardrail.pt` | PyTorch weights |
| `tokenizer/` | HuggingFace tokenizer |
| `mahalanobis_stats.npz` | OOD mean / inverse covariance |

---

## Export ONNX

```bash
python export_onnx.py \
  --model-dir ./outputs/minilm_run \
  --out ./outputs/guardrail_model.onnx
```

Inputs: `input_ids`, `attention_mask`  
Outputs: `logits`, `embeddings`

### Quantization (INT8)

Quantize with ONNX Runtime tools or your pipeline, then save as:

```
safellmkit-ml/src/commonMain/resources/assets/guardrail_model_int8.onnx
```

The JVM runtime loads `guardrail_model_int8.onnx` by default.

Copy Mahalanobis arrays:

```python
import numpy as np
d = np.load("outputs/minilm_run/mahalanobis_stats.npz")
np.save("../safellmkit-ml/src/commonMain/resources/assets/mean.npy", d["mean"])
np.save("../safellmkit-ml/src/commonMain/resources/assets/inv_cov.npy", d["inv_cov"])
```

Copy tokenizer:

```bash
cp -r outputs/minilm_run/tokenizer/* ../safellmkit-ml/src/commonMain/resources/assets/tokenizer/
```

---

## Train temporal BiLSTM (Crescendo / PAIR)

Script: `train_temporal_bilstm.py`

- Input: sequences of shape `[batch, 5, 384]` (last 5 MiniLM embeddings)
- Output: 3 classes — SAFE, PAIR, CRESCENDO
- Loss: Focal loss
- Export: `outputs/temporal_model.onnx`

```bash
python train_temporal_bilstm.py
```

Copy to runtime:

```bash
cp outputs/temporal_model.onnx ../safellmkit-ml/src/commonMain/resources/assets/
```

> The stock script uses synthetic data for plumbing validation. Replace `dummy_x` / `dummy_y` with real multi-turn embedding sequences for production quality.

---

## Other scripts

| Script | Purpose |
|--------|---------|
| `train_guardrail.py` | Earlier guardrail training variant |
| `train_jailbreak_onnx.py` | Legacy jailbreak exporter |
| `train_jailbreak_onnx_from_csv.py` | CSV-based legacy pipeline |
| `verify_onnx.py` | Sanity-check exported ONNX |
| `focal_loss.py` | Shared focal + triplet helpers |

---

## End-to-end workflow

```bash
# 1. Prepare data
python build_dataset.py --input ./raw --out ./data/train.jsonl

# 2. Train MiniLM + Mahalanobis
python train_minilm_guardrail.py --train ./data/train.jsonl --val ./data/val.jsonl --out ./outputs/run1

# 3. Export ONNX
python export_onnx.py --model-dir ./outputs/run1 --out ./outputs/guardrail_model.onnx

# 4. Train temporal model
python train_temporal_bilstm.py

# 5. Copy assets into safellmkit-ml (see safellmkit-ml/README.md)

# 6. Verify runtime
cd .. && ./gradlew :safellmkit-core:jvmTest --tests "com.safellmkit.core.engine.GuardrailEngineTest.testRealModelInference"
```

---

## Related

- [safellmkit-ml/README.md](../safellmkit-ml/README.md) — runtime assets
- [docs/ML_INTEGRATION.md](../docs/ML_INTEGRATION.md)
