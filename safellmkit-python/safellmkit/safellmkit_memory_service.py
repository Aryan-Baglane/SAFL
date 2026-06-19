from fastapi import FastAPI
from pydantic import BaseModel
import time
from embedding import EmbeddingProvider
from redis_store import RedisStore

app = FastAPI(title="SafeLLMKit Memory Service")

store = RedisStore()
embedder = EmbeddingProvider()


class TurnIn(BaseModel):
    session_id: str
    turn_index: int
    role: str = "user"
    content: str
    timestamp_ms: int | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/session/turn")
def add_turn(payload: TurnIn):
    ts = payload.timestamp_ms or int(time.time() * 1000)
    embedding = embedder.embed(payload.content)

    turn = {
        "session_id": payload.session_id,
        "turn_index": payload.turn_index,
        "role": payload.role,
        "content": payload.content,
        "timestamp_ms": ts,
        "embedding": embedding,
    }

    store.append_turn(payload.session_id, turn)
    store.set_meta(
        payload.session_id,
        last_turn_index=payload.turn_index,
        updated_at_ms=ts
    )

    return {"ok": True, "embedding_dim": len(embedding)}


@app.get("/session/{session_id}/recent")
def recent(session_id: str, limit: int = 8):
    return store.get_recent_turns(session_id, limit)


@app.get("/session/{session_id}/state")
def state(session_id: str):
    return store.get_state(session_id) or {
        "session_id": session_id,
        "status": "UNKNOWN"
    }