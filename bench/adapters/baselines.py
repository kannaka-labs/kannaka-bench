"""The two honest floors every memory system must beat.

recency — no retrieval at all: the k most recent items. This is what "plain
context" does for recall@k (a fixed window keeps the tail of the history);
phase 2 gives the answer model the whole history instead.

vector_numpy — sentence-transformers all-MiniLM-L6-v2 (the same encoder
kannaka's default hash->minilm flip uses, so the comparison is about the
medium, not the embedding) and cosine similarity over a numpy matrix. No
index, no ANN: exact top-k, which is the strongest thing a vector baseline
can do at this scale.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit, dir_bytes


class RecencyAdapter(Adapter):
    name = "recency"

    def __init__(self):
        self.items: list[MemoryItem] = []
        self.dir = None

    def open(self, run_dir: str) -> None:
        self.items = []
        self.dir = os.path.join(run_dir, "recency")
        os.makedirs(self.dir, exist_ok=True)

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        for it in items:
            self.items.append(it)
        with open(os.path.join(self.dir, "items.jsonl"), "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps({"id": it.id, "text": it.text}) + "\n")

    def recall(self, query: str, k: int, when=None) -> list[RecallHit]:
        tail = self.items[-k:][::-1]
        return [RecallHit(id=it.id, score=1.0 - i / max(1, k), text=it.text[:200]) for i, it in enumerate(tail)]

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir) if self.dir else 0


class VectorNumpyAdapter(Adapter):
    name = "vector_numpy"
    MODEL = os.environ.get("BENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    def __init__(self):
        self.model = None
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.M = None
        self.dir = None

    def open(self, run_dir: str) -> None:
        import numpy as np  # noqa: F401
        from sentence_transformers import SentenceTransformer
        if self.model is None:
            self.model = SentenceTransformer(self.MODEL)
        self.ids, self.texts, self.M = [], [], None
        self.dir = os.path.join(run_dir, "vector_numpy")
        os.makedirs(self.dir, exist_ok=True)

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        import numpy as np
        batch = list(items)
        if not batch:
            return
        emb = self.model.encode([it.text for it in batch], normalize_embeddings=True,
                                batch_size=64, show_progress_bar=False)
        self.ids += [it.id for it in batch]
        self.texts += [it.text for it in batch]
        self.M = emb if self.M is None else np.vstack([self.M, emb])
        np.save(os.path.join(self.dir, "embeddings.npy"), self.M)
        with open(os.path.join(self.dir, "ids.json"), "w") as f:
            json.dump(self.ids, f)

    def recall(self, query: str, k: int, when=None) -> list[RecallHit]:
        import numpy as np
        if self.M is None or not len(self.ids):
            return []
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.M @ q
        top = np.argsort(-scores)[:k]
        return [RecallHit(id=self.ids[i], score=float(scores[i]), text=self.texts[i][:200]) for i in top]

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir) if self.dir else 0


class VectorBgeAdapter(VectorNumpyAdapter):
    """Exact cosine over BAAI/bge-base-en-v1.5 (768-d) — the encoder row that
    pairs with `kannaka_bge`; same weights, in-process."""
    name = "vector_bge"
    MODEL = "BAAI/bge-base-en-v1.5"
