"""Mem0 (mem0ai) as a bench adapter — the first non-kannaka *memory* row.

Mem0 differs from every other adapter here in kind, not degree. `vector_numpy`
and `kannaka_*` store the turns they are given; Mem0 runs an **LLM over each
turn and stores extracted facts** — rewritten sentences, not the original text.
That is the thing being measured, and it has three consequences the row has to
carry honestly:

1. **Attribution.** A hit is scored against the gold TURN id, but Mem0 returns
   its own rewritten memory. The only way to score it is to attach
   `item_id` to each add() and read it back off the search result's metadata —
   verified to survive the round trip before this adapter was written. It
   follows that turns must be added ONE AT A TIME: batching would blur several
   turns into one extracted fact carrying one arbitrary `item_id`, and every
   hit against the others would read as a miss.

2. **Cost.** One LLM call per ingested turn. A LongMemEval question is ~500
   turns, so a 30-question run is ~15,000 calls. Every other adapter here
   ingests for free. `llm_calls` is reported so the table can say so.

3. **Recall is lossy by design.** Extraction can drop a turn entirely — no
   fact, no memory, no possible hit. `dropped_turns` counts those, because a
   miss caused by ingest is a different finding from a miss caused by ranking.

Config notes, each one a bug found while wiring this up:
  * mem0's own ollama embedder calls `client.list()[i].get("model")` and every
    modern ollama-python returns pydantic objects, so it dies with
    `argument of type 'NoneType' is not iterable`. We use the OpenAI-shaped
    endpoint of the bench's embed_server instead — which is also what keeps
    the encoder control exact: Mem0 gets the SAME all-MiniLM weights as
    `vector_numpy` and `kannaka_minilm`, so the row is about memory
    architecture and not about embeddings.
  * qdrant is pointed at an on-disk path under the run dir. The bench opens one
    store PER QUESTION; a shared server would leak memories between questions.
  * ⚠⚠ ONE MEM0 RUN PER **HOME**. Besides the per-store qdrant configured
    here, mem0 opens a GLOBAL embedded qdrant at ~/.mem0/migrations_qdrant,
    and embedded qdrant takes an exclusive lock on its folder. A second bench
    process — even one pointed at a different --out — dies with "Storage
    folder ... is already accessed by another instance of Qdrant client", and
    the bench records it as a per-question `error` row rather than a crash,
    so it looks like an adapter fault instead of contention.
    **Giving each worker its own `HOME` lifts this** — VERIFIED 2026-09-21,
    6 concurrent workers, zero lock errors. ⚠ But `HOME` is load-bearing for
    more than mem0: it also relocates `pip install --user` site-packages and
    every `~`-relative path, so a worker launched that way needs `PYTHONPATH`
    and any repo root passed absolutely. Both bit this calibration.
  * Extraction runs on a FREE LOCAL model (qwen2.5:7b via ollama), not on the
    Claude gateway. MEASURED: Claude cost $0.019/turn — $20 bought ~1,034 turns
    (two questions), and a full 30-question run would be ~$280. Report the
    extraction model beside any mem0 number: it is part of the configuration
    under test, and a Mem0 user on GPT-4 would get different extraction.
  * Running locally also lets temperature be pinned to 0, so extraction is
    deterministic. Against Anthropic it could not be: mem0 sends temperature
    AND top_p together, Anthropic rejects that, and the only escape
    (`is_reasoning_model`) strips temperature entirely.
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit, dir_bytes

USER = "bench"


class Mem0Adapter(Adapter):
    name = "mem0"

    def __init__(self):
        self.dir = None
        self.m = None
        self.llm_calls = 0
        self.dropped_turns = 0
        self.ingest_errors = 0
        self.ingest_retries = 0

    # -- config ---------------------------------------------------------
    def _config(self, store: str) -> dict:
        # Extraction defaults to a FREE, NEUTRAL local model. Measured
        # 2026-09-21: Claude (agent-brain) cost $0.019 per turn -> $20 bought
        # ~1,034 turns, i.e. two questions, and a full 30-question run would be
        # ~$280. Not affordable, and not necessary: mem0 only needs a model
        # competent at "pull the facts out of this turn".
        #
        # qwen2.5:7b is deliberately NOT kannaka-brain. Using our own model to
        # power a competitor's ingest would make any weak mem0 row arguable as
        # our model's fault rather than mem0's.
        llm_url = os.environ.get("BENCH_MEM0_LLM_URL", "http://172.18.0.1:11434/v1")
        llm_key = os.environ.get("BENCH_MEM0_LLM_KEY", "ollama")
        embed = os.environ.get("BENCH_EMBED_URL", "http://127.0.0.1:11437/v1")
        model = os.environ.get("BENCH_MEM0_LLM", "qwen2.5:7b")
        dims = int(os.environ.get("BENCH_MEM0_DIMS", "384"))
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {"collection_name": "bench", "path": store,
                           "on_disk": True, "embedding_model_dims": dims},
            },
            # temperature pinned: only Anthropic rejects temperature+top_p
            # together, so with a local model extraction can be deterministic.
            # (`is_reasoning_model` was the Anthropic workaround and cost us
            # that determinism; it is no longer needed.)
            "llm": {
                "provider": "openai",
                "config": {"model": model, "openai_base_url": llm_url,
                           "api_key": llm_key, "temperature": 0.0,
                           "max_tokens": 1024},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": os.environ.get("BENCH_MEM0_EMBED_MODEL", "all-minilm"),
                           "openai_base_url": embed, "api_key": "unused",
                           "embedding_dims": dims},
            },
        }

    def open(self, run_dir: str) -> None:
        from mem0 import Memory  # imported late: only this adapter needs mem0ai

        self.dir = os.path.join(run_dir, "mem0")
        # A stale store would silently answer another question's queries.
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        self.llm_calls = self.dropped_turns = self.ingest_errors = self.ingest_retries = 0
        self.m = Memory.from_config(self._config(self.dir))

    # -- ingest ---------------------------------------------------------
    def ingest(self, items: Iterable[MemoryItem]) -> None:
        for it in items:
            text = (it.text or "").strip()
            if not text:
                continue
            meta = {"item_id": it.id}
            if it.when is not None:
                try:
                    meta["when"] = it.when.astimezone(timezone.utc).isoformat()
                except (AttributeError, ValueError):
                    pass
            # One retry. MEASURED on a 4090 2026-09-21: ~4% of turns
            # (13 of 300 across three runs) die inside mem0 with
            # `AttributeError: 'str' object has no attribute 'get'` — the
            # extractor returned JSON that parsed to a str where mem0 expects
            # a dict, and mem0 has no guard. Extraction is sampled, so the
            # same turn usually succeeds on a second attempt.
            #
            # This is a FAIRNESS fix, not a cosmetic one. A turn lost at
            # ingest can never be recalled, so 4% loss is 4% of the gold
            # evidence deleted before ranking is even tested — it would show
            # up in our table as Mem0 ranking badly when in fact our harness
            # threw the turn away. Retries are counted separately so the
            # instability stays visible instead of being papered over.
            err = None
            for attempt in (1, 2):
                try:
                    r = self.m.add([{"role": "user", "content": text[:4000]}],
                                   user_id=USER, metadata=meta)
                    self.llm_calls += 1
                    if attempt == 2:
                        self.ingest_retries += 1
                    got = r.get("results", []) if isinstance(r, dict) else (r or [])
                    if not got:
                        # Extraction found nothing worth keeping. Not an error,
                        # but this turn can never be recalled, so it is counted.
                        self.dropped_turns += 1
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = e
                    self.llm_calls += 1   # the call was spent either way
            if err is not None:
                # One bad turn must not lose the question, but a silent skip
                # would understate ingest loss — count and say so. The message
                # is NOT truncated to 140 chars: that truncation once hid an
                # out-of-credit error behind `{"type":"error","error":{"type`.
                self.ingest_errors += 1
                if self.ingest_errors <= 3:
                    print(f"[mem0] add failed twice for {it.id}: {type(err).__name__}: {err}",
                          file=sys.stderr, flush=True)

    # -- recall ---------------------------------------------------------
    def recall(self, query: str, k: int, when: datetime | None = None) -> list[RecallHit]:
        try:
            s = self.m.search(query[:2000], filters={"user_id": USER}, limit=max(k, 1))
        except Exception as e:  # noqa: BLE001
            print(f"[mem0] search failed: {type(e).__name__}: {str(e)[:140]}",
                  file=sys.stderr, flush=True)
            return []
        hits = s.get("results", []) if isinstance(s, dict) else (s or [])
        out, seen = [], set()
        for h in hits:
            md = h.get("metadata") or {}
            iid = md.get("item_id")
            if not iid:
                # A memory Mem0 merged or created without our metadata. It
                # cannot be scored against a gold turn, but dropping it
                # silently would flatter the row by shortening the list.
                iid = f"mem0:{h.get('id')}"
            if iid in seen:
                continue
            seen.add(iid)
            out.append(RecallHit(id=iid, score=float(h.get("score") or 0.0),
                                 text=str(h.get("memory") or "")[:200]))
        return out[:k]

    def ingest_stats(self) -> dict:
        """What ingest cost and lost — reported in the manifest.

        Every other adapter ingests for free and loses nothing, so these
        columns exist to stop Mem0's numbers being read as like-for-like.
        """
        return {"llm_calls": self.llm_calls,
                "dropped_turns": self.dropped_turns,
                "ingest_errors": self.ingest_errors,
                "ingest_retries": self.ingest_retries}

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir) if self.dir else 0

    def close(self) -> None:
        self.m = None
