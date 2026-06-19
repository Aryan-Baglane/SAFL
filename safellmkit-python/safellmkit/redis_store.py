import json
import numpy as np
import redis
from typing import Any, Optional

class RedisStore:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.index_name = "guardrail_turn_idx"
        self.prefix = "guardrail:turn:"

    def session_turns_key(self, session_id: str) -> str:
        return f"guardrail:session:{session_id}:turns"

    def session_state_key(self, session_id: str) -> str:
        return f"guardrail:session:{session_id}:state"

    def append_turn(self, session_id: str, turn: dict[str, Any], embedding: list[float]):
        turn_index = int(turn.get("turn_index", 0))
        key = f"{self.prefix}{session_id}:{turn_index}"
        emb = np.asarray(embedding, dtype=np.float32)

        payload = {
            "session_id": session_id,
            "turn_index": turn_index,
            "role": turn.get("role", "user"),
            "content": turn.get("content", ""),
            "embedding": emb.tobytes(),
            "embedding_dim": emb.shape[0],
        }

        pipe = self.client.pipeline()
        pipe.hset(key, mapping=payload)
        pipe.rpush(self.session_turns_key(session_id), key.encode("utf-8"))
        pipe.execute()

    def get_recent_turns(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        keys = self.client.lrange(self.session_turns_key(session_id), max(0, -limit), -1)
        out = []
        for key in keys:
            raw = self.client.hgetall(key)
            if not raw:
                continue
            out.append(self._decode_hash(raw))
        return out

    def set_state(self, session_id: str, state: dict[str, Any]) -> None:
        self.client.set(self.session_state_key(session_id), json.dumps(state).encode("utf-8"))

    def get_state(self, session_id: str) -> Optional[dict[str, Any]]:
        raw = self.client.get(self.session_state_key(session_id))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _decode_hash(self, raw: dict[bytes, bytes]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for k, v in raw.items():
            key = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
            if key in {"turn_index", "embedding_dim"}:
                decoded[key] = int(v)
            elif key == "embedding":
                decoded[key] = np.frombuffer(v, dtype=np.float32).tolist()
            else:
                decoded[key] = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
        return decoded

    def ensure_vector_index(self, dim: int = 384) -> None:
        """
        Best effort Redis Stack vector index.
        If Redis Stack/RediSearch is unavailable, the app still works.
        """
        try:
            self.client.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", self.prefix,
                "SCHEMA",
                "embedding", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", str(dim),
                "DISTANCE_METRIC", "COSINE"
            )
        except Exception:
            pass

    def vector_search(self, query_embedding: list[float], k: int = 5) -> list[dict[str, Any]]:
        """
        Try RediSearch KNN. If unavailable, fallback to a simple scan over recent keys.
        """
        q = np.asarray(query_embedding, dtype=np.float32)
        try:
            blob = q.tobytes()
            res = self.client.execute_command(
                "FT.SEARCH", self.index_name,
                f"(*)=>[KNN {k} @embedding $vec AS score]",
                "PARAMS", "2", "vec", blob,
                "SORTBY", "score",
                "RETURN", "5", "session_id", "turn_index", "content", "score", "embedding_dim",
                "DIALECT", "2"
            )
            return [{"raw": str(res)}]
        except Exception:
            scored = []
            for key in self.client.scan_iter(match=f"{self.prefix}*"):
                raw = self.client.hgetall(key)
                if not raw:
                    continue
                item = self._decode_hash(raw)
                emb = np.asarray(item.get("embedding", []), dtype=np.float32)
                if emb.size != q.size or emb.size == 0:
                    continue
                denom = (np.linalg.norm(emb) * np.linalg.norm(q)) + 1e-8
                score = float(np.dot(emb, q) / denom)
                item["score"] = score
                scored.append(item)
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:k]