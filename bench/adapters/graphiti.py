"""Graphiti (getzep/graphiti, Apache-2.0) — the open-source engine behind Zep.

  BENCH_GRAPHITI_LLM_URL   http://172.18.0.1:11434/v1   (any OpenAI-compatible endpoint)
  BENCH_GRAPHITI_LLM       qwen2.5:7b
  BENCH_GRAPHITI_LLM_KEY   ollama
  BENCH_EMBED_URL          http://127.0.0.1:11437/v1    (the bench embed server)

Why Graphiti and not "Zep": Zep's product is a hosted service; Zep Community
Edition was deprecated, and what Zep open-sources is Graphiti, the temporal
knowledge-graph engine its memory is built on. This adapter measures
graphiti-core with an embedded Kuzu graph (one database directory per store —
no server, nothing shared between questions).

What Graphiti does per ingested turn (`add_episode`): several LLM calls —
extract entities, deduplicate them against the graph, extract relations
("facts", i.e. edges), resolve/invalidate contradicting facts, summarise
nodes. Like Mem0, and unlike every retrieval-only row, **ingest runs an LLM
over every turn**; `llm_calls` and tokens are counted from the wire (the
generic client does not record usage itself, so the adapter wraps
`chat.completions.create`) and reported so the row is not read as
like-for-like.

Attribution: each turn is one episode NAMED by its item id; Graphiti's search
returns facts (EntityEdge), each carrying the uuids of the episodes it was
extracted from. A fact expands to its source turns in order, deduplicated,
cut to k. A fact merged from several turns therefore credits all of them —
the fair reading of a system that consolidates. A turn from which no fact was
extracted can never be recalled: counted as `dropped_turns`.

Search: `Graphiti.search()` — the documented default (hybrid BM25 + cosine
over facts, reciprocal-rank fusion; no cross-encoder, so no LLM at recall).
Encoder held fixed: `OpenAIEmbedder` pointed at the bench embed server
(all-MiniLM-L6-v2, 384-d — the same weights as every other MiniLM row).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit, dir_bytes


class _Meter:
    """Counts LLM calls and tokens as they cross the wire."""

    def __init__(self):
        self.calls = self.prompt = self.completion = self.errors = 0

    def wrap(self, client):
        orig = client.chat.completions.create
        meter = self

        async def create(*a, **kw):
            meter.calls += 1
            try:
                r = await orig(*a, **kw)
            except Exception:
                meter.errors += 1
                raise
            u = getattr(r, "usage", None)
            if u is not None:
                meter.prompt += int(getattr(u, "prompt_tokens", 0) or 0)
                meter.completion += int(getattr(u, "completion_tokens", 0) or 0)
            return r
        client.chat.completions.create = create
        return client


class GraphitiAdapter(Adapter):
    name = "graphiti"

    def __init__(self):
        self.dir = None
        self.g = None
        self.loop = None
        self.meter = _Meter()
        self.ep_item: dict[str, str] = {}
        self.version = None
        self._reset()

    def _reset(self):
        self.meter = _Meter()
        self.ep_item = {}
        self.dropped_turns = 0
        self.ingest_errors = 0
        self.turns = 0
        self.facts = 0

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def open(self, run_dir: str) -> None:
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
        from openai import AsyncOpenAI
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        try:
            from importlib.metadata import version
            self.version = version("graphiti-core")
        except Exception:  # noqa: BLE001
            self.version = "unknown"
        self._reset()
        self.dir = os.path.join(run_dir, "graphiti")
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        if self.loop is None:
            self.loop = asyncio.new_event_loop()

        llm_url = os.environ.get("BENCH_GRAPHITI_LLM_URL", "http://172.18.0.1:11434/v1")
        llm_key = os.environ.get("BENCH_GRAPHITI_LLM_KEY", "ollama")
        model = os.environ.get("BENCH_GRAPHITI_LLM", "qwen2.5:7b")
        cfg = LLMConfig(api_key=llm_key, model=model, small_model=model, base_url=llm_url, temperature=0.0)
        raw = self.meter.wrap(AsyncOpenAI(api_key=llm_key, base_url=llm_url, timeout=900.0))
        llm = OpenAIGenericClient(config=cfg, client=raw)
        emb = OpenAIEmbedder(OpenAIEmbedderConfig(
            api_key="unused", base_url=os.environ.get("BENCH_EMBED_URL", "http://127.0.0.1:11437/v1"),
            embedding_model=os.environ.get("BENCH_GRAPHITI_EMBED_MODEL", "all-minilm"),
            embedding_dim=int(os.environ.get("BENCH_GRAPHITI_DIMS", "384"))))
        # Never called: Graphiti.search() uses RRF, no cross-encoder. Passed so
        # the constructor does not build its default OpenAI reranker.
        rer = OpenAIRerankerClient(config=cfg, client=raw)
        driver = KuzuDriver(db=os.path.join(self.dir, "graph.kuzu"))
        self.g = Graphiti(graph_driver=driver, llm_client=llm, embedder=emb, cross_encoder=rer)
        self._run(self.g.build_indices_and_constraints())

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        from graphiti_core.nodes import EpisodeType
        for it in items:
            text = (it.text or "").strip()
            if not text:
                continue
            self.turns += 1
            when = it.when or datetime.now(timezone.utc)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            err = None
            for attempt in (1, 2):   # one retry, counted, as for Mem0
                try:
                    r = self._run(self.g.add_episode(
                        name=it.id, episode_body=text[:4000], source=EpisodeType.message,
                        source_description="conversation turn", reference_time=when))
                    self.ep_item[r.episode.uuid] = it.id
                    self.facts += len(r.edges)
                    if not r.edges:
                        self.dropped_turns += 1
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = e
            if err is not None:
                self.ingest_errors += 1
                if self.ingest_errors <= 3:
                    print(f"[graphiti] add_episode failed twice for {it.id}: {type(err).__name__}: {err}",
                          file=sys.stderr, flush=True)

    def recall(self, query: str, k: int, when=None) -> list[RecallHit]:
        try:
            edges = self._run(self.g.search(query[:2000], num_results=max(k, 1) * 3))
        except Exception as e:  # noqa: BLE001
            print(f"[graphiti] search failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
            return []
        out, seen = [], set()
        for rank, e in enumerate(edges):
            srcs = [self.ep_item.get(u) or f"graphiti:{u}" for u in (getattr(e, "episodes", None) or [])]
            if not srcs:
                srcs = [f"graphiti:{getattr(e, 'uuid', rank)}"]
            for iid in srcs:
                if iid in seen:
                    continue
                seen.add(iid)
                out.append(RecallHit(id=iid, score=1.0 - rank / max(1, len(edges)),
                                     text=str(getattr(e, "fact", ""))[:200]))
                if len(out) == k:
                    return out
        return out

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir) if self.dir else 0

    def ingest_stats(self) -> dict:
        """Numeric only (run.py sums these across stores)."""
        return {"llm_calls": self.meter.calls, "prompt_tokens": self.meter.prompt,
                "completion_tokens": self.meter.completion, "llm_errors": self.meter.errors,
                "dropped_turns": self.dropped_turns, "ingest_errors": self.ingest_errors,
                "facts": self.facts, "turns": self.turns}

    def describe(self) -> dict:
        return {"graphiti_core": self.version, "graph": "kuzu (embedded, one db per store)",
                "llm": os.environ.get("BENCH_GRAPHITI_LLM", "qwen2.5:7b"),
                "search": "Graphiti.search (EDGE_HYBRID_SEARCH_RRF)", "embedder": "all-minilm 384 via bench embed server"}

    def close(self) -> None:
        if self.g is not None:
            try:
                self._run(self.g.close())
            except Exception:  # noqa: BLE001
                pass
        self.g = None
