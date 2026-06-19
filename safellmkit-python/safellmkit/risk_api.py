import os
import json
import math
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from embedding import EmbeddingProvider
from redis_store import RedisStore
from perplexity_scorer import GPT2PerplexityScorer
from smoothllm import SmoothLLMEnsemble

try:
    import onnxruntime as ort
except Exception:
    ort = None

app = FastAPI(title="SafeLLMKit Risk API")

store = RedisStore(os.getenv("SAFE_LLMKIT_REDIS_URL", "redis://localhost:6379/0"))
embedder = EmbeddingProvider(os.getenv("SAFE_LLMKIT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
perplexity = GPT2PerplexityScorer(os.getenv("SAFE_LLMKIT_PPL_MODEL", "gpt2"))

class InspectIn(BaseModel):
    session_id: str
    turn_index: int
    role: str = "user"
    content: str

class OnnxClassifier:
    def __init__(self, model_path: Optional[str]):
        self.session = None
        self.model_path = model_path
        if model_path and ort is not None:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def probability(self, text: str) -> Optional[float]:
        return None

onnx_model = OnnxClassifier(os.getenv("SAFE_LLMKIT_ONNX_MODEL"))

def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def heuristic_risk(text: str) -> float:
    t = text.lower()
    score = 0.0
    if any(k in t for k in ["ignore previous instructions", "developer mode", "jailbreak", "system prompt", "roleplay"]):
        score += 0.35
    if any(k in t for k in ["reveal your", "reveal the", "bypass safety", "do anything now"]):
        score += 0.20
    if any(k in t for k in ["@", "password", "credit card", "ssn", "otp"]):
        score += 0.15
    weird_chars = sum(1 for ch in text if ord(ch) < 32 or ch in ["\u200b", "\u200c", "\u200d", "\u2060"])
    score += min(0.20, weird_chars / max(1, len(text)))
    repetition = 1.0 - (len(set(t.split())) / max(1, len(t.split())))
    score += min(0.10, repetition)
    return max(0.0, min(1.0, score))

def normalize_perplexity(pp: float) -> float:
    if pp <= 1.0:
        return 0.0
    low = math.log(10.0)
    high = math.log(500.0)
    return max(0.0, min(1.0, (math.log(pp) - low) / (high - low)))

def smooth_score(text: str) -> dict:
    def base_predict(t: str) -> float:
        # Local fallback. If you wire ONNX later, call that here.
        return heuristic_risk(t)

    ensemble = SmoothLLMEnsemble(base_predict, trials=12)
    r = ensemble.score(text, threshold=0.5)
    return {
        "mean_score": r.mean_score,
        "variance": r.variance,
        "attack_vote_ratio": r.attack_vote_ratio,
    }

def mahalanobis_score(embedding: list[float]) -> float:
    stats_path = os.getenv("SAFE_LLMKIT_MAHALANOBIS_STATS", "")
    if not stats_path or not os.path.exists(stats_path):
        return 0.0
    data = np.load(stats_path)
    mean = data["mean"].astype(np.float32)
    inv_cov = data["inv_cov"].astype(np.float32)
    x = np.asarray(embedding, dtype=np.float32)
    n = min(len(x), len(mean))
    diff = x[:n] - mean[:n]
    inv = inv_cov[:n, :n]
    val = float(np.sqrt(max(0.0, diff.T @ inv @ diff)))
    return val

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/inspect")
def inspect(inp: InspectIn):
    embedding = embedder.embed(inp.content)
    recent = store.get_recent_turns(inp.session_id, limit=8)

    if recent:
        embeddings = [x.get("embedding") for x in recent if x.get("embedding")]
        if embeddings:
            centroid = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0)
            semantic_similarity = _cosine(centroid, embedding)
        else:
            semantic_similarity = 1.0
    else:
        semantic_similarity = 1.0

    semantic_drift = 1.0 - semantic_similarity
    h = heuristic_risk(inp.content)
    pp = perplexity.perplexity(inp.content)
    ppl_score = normalize_perplexity(pp)

    smooth = smooth_score(inp.content)
    maha = mahalanobis_score(embedding)
    ml_prob = onnx_model.probability(inp.content)

    risk = (
        h * 0.26 +
        semantic_drift * 0.18 +
        ppl_score * 0.18 +
        smooth["variance"] * 0.12 +
        min(1.0, maha / 10.0) * 0.12 +
        (ml_prob or 0.0) * 0.14
    )
    risk = max(0.0, min(1.0, risk))

    action = "ALLOW"
    if risk >= 0.80:
        action = "BLOCK"
    elif risk >= 0.55:
        action = "WARN"

    result = {
        "session_id": inp.session_id,
        "turn_index": inp.turn_index,
        "risk_score": risk,
        "action": action,
        "semantic_similarity": semantic_similarity,
        "semantic_drift": semantic_drift,
        "perplexity": pp,
        "perplexity_score": ppl_score,
        "heuristic_risk": h,
        "smooth_mean_score": smooth["mean_score"],
        "smooth_variance": smooth["variance"],
        "smooth_attack_vote_ratio": smooth["attack_vote_ratio"],
        "mahalanobis_score": maha,
        "ml_probability": ml_prob,
    }

    store.append_turn(
        inp.session_id,
        {
            "turn_index": inp.turn_index,
            "role": inp.role,
            "content": inp.content,
        },
        embedding=embedding
    )
    store.set_state(inp.session_id, result)
    return result

@app.get("/session/{session_id}/recent")
def recent(session_id: str, limit: int = 8):
    return store.get_recent_turns(session_id, limit)

@app.get("/session/{session_id}/state")
def state(session_id: str):
    return store.get_state(session_id) or {"session_id": session_id, "status": "UNKNOWN"}