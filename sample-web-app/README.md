# SafeLLMKit Web Demo

Light-themed **AI Firewall Console** demonstrating enforced guardrail gating with live SDK telemetry.

## Features

- **3-column dashboard** — config/chat, pipeline telemetry, session memory
- **Custom OpenRouter models** — search presets or type any model ID (e.g. `cohere/north-mini-code:free`)
- **Provider call counter** — proves blocked prompts never reach OpenRouter
- **Pipeline rail** — INT → HEUR → ONNX → LSTM → VSIM → AGGR animation
- **Perplexity & SmoothLLM charts** — derived from real `safellmkit-js` inspection results
- **Redis-style state machine, centroid drift, SDK-aligned reputation decay preview**

## Run

```bash
cd ../safellmkit-js && npm install
cd ../sample-web-app && npm install && npm run dev
```

Open http://localhost:5173

## Custom model

1. Open the **Target model endpoint** dropdown
2. Type any OpenRouter model slug (e.g. `meta-llama/llama-3-70b-instruct`)
3. Click **+ Use custom** or **Apply**

## Enforcement flow

```
prompt → safellmkit-js (input) → session memory → BLOCK? → stop
      → OpenRouter (only if allowed) → output scan → display
```

Toggle **SDK GUARD OFF** to compare bypass behavior.

## SDK-aligned updates

- Session reputation now follows the SDK’s saturating accumulation model instead of recovering every turn.
- Reputation decay is shown as an inactivity-based preview (`6h` gate, then decay), matching the latest SDK behavior.
- Blocked turns are recorded once in session memory, fixing prior over-counting in the demo escalation path.
