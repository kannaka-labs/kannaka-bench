"""Letta (formerly MemGPT) through its self-hosted V1 API server.

  BENCH_LETTA_URL        http://127.0.0.1:8283    (docker letta/letta:0.16.8)
  BENCH_LETTA_LLM        ollama/qwen2.5:7b        (agent arm only; the model handle)
  BENCH_LETTA_CTX        agent arm: context_window_limit (match the ollama server's context)
  BENCH_EMBED_URL        http://127.0.0.1:11437/v1 (the bench embed server, OpenAI shape)
  BENCH_LETTA_PG_DSN     optional: the server's Postgres, read-only, for footprint
  BENCH_LETTA_MAX_STEPS  agent arm: max agent steps per turn (default: the server's)

⚠ WHICH LETTA. The V1 API server — MemGPT's archival memory, recall memory and
agent-driven memory edits — was RETIRED in 2026: its last release is 0.16.8
(2026-05-14, `letta/letta:0.16.8`, source on the `archive` branch of
letta-ai/letta). Since then `pip install letta` and `letta/letta:latest` are
**Letta Code**, whose memory is a git-backed markdown filesystem the agent
edits and greps; it has no ranked-retrieval API, so there is nothing to score
by item id. This adapter measures the last self-hostable Letta memory server;
the row must say so.

Two arms, because Letta has two ways memory gets written:

  `letta_archival`  every turn inserted into the agent's archival memory with
                    `passages.create(text, tags=[item_id], created_at=when)`,
                    recalled with `passages.search` (the API the agent's own
                    archival_memory_search tool uses). No LLM. Letta chunks a
                    passage at its default `embedding_chunk_size` (300) — a
                    feature of the system, kept; every chunk carries the tag.
  `letta_agent`     every turn is SENT TO THE AGENT as a user message; the
                    agent's LLM decides what to write (archival inserts, core
                    memory edits) — MemGPT's actual design. One LLM call per
                    agent step, several steps per turn. Attribution: turns are
                    sent one at a time, so a passage that appears while turn X
                    is being processed is attributed to X (the same honesty
                    rule as Mem0's item_id: a rewritten fact still points at
                    the turn it came from). Recall = archival search over what
                    the agent chose to keep. Turns it chose NOT to archive can
                    never be recalled; they are counted (`dropped_turns`).

Encoder held fixed: the agent's embedding_config points at the bench embed
server's OpenAI-shaped endpoint (all-MiniLM-L6-v2, 384-d — the same weights as
vector_numpy/kannaka_minilm/pgvector). Letta pads every vector to 4096-d in
Postgres (vector(4096)) and has no vector index on archival passages: search
is an exact scan filtered by archive. Both are its storage design, measured.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import timezone
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit

URL = os.environ.get("BENCH_LETTA_URL", "http://127.0.0.1:8283")


def embedding_config() -> dict:
    return {
        "embedding_endpoint_type": "openai",
        "embedding_endpoint": os.environ.get("BENCH_EMBED_URL", "http://127.0.0.1:11437/v1"),
        "embedding_model": os.environ.get("BENCH_LETTA_EMBED_MODEL", "all-minilm"),
        "embedding_dim": int(os.environ.get("BENCH_LETTA_DIMS", "384")),
        # Letta's own default chunk size; not tuned.
        "embedding_chunk_size": int(os.environ.get("BENCH_LETTA_CHUNK", "300")),
    }


def llm_kwargs() -> dict:
    """The agent's model: a handle the server resolves (BENCH_LETTA_LLM).
    `ollama/<model>` uses the server's OLLAMA_BASE_URL; an LLM on another box
    is registered once as a named provider (`POST /v1/providers/`
    {"name": "podollama", "provider_type": "ollama", "base_url": ...}) and
    addressed as `podollama/<model>`. (An explicit `llm_config` is rejected by
    server 0.16.8 with a model_settings discriminator error.)
    BENCH_LETTA_CTX sets `context_window_limit` so Letta's idea of the window
    matches what the ollama server actually serves (OLLAMA_CONTEXT_LENGTH /
    num_ctx) — otherwise Letta assumes the model's native 32k and ollama
    silently truncates the prompt."""
    kw = {"model": os.environ.get("BENCH_LETTA_LLM", "ollama/qwen2.5:7b")}
    if os.environ.get("BENCH_LETTA_CTX"):
        kw["context_window_limit"] = int(os.environ["BENCH_LETTA_CTX"])
    return kw


class LettaArchivalAdapter(Adapter):
    name = "letta_archival"
    agent_driven = False

    def __init__(self, client=None):
        self._client = client          # injectable for the unit suite
        self.agent_id = None
        self.passage_item: dict[str, str] = {}   # passage id -> item id (agent arm)
        self.known: set[str] = set()
        self.tools: list[str] = []
        self.server_version = None
        self._reset_counters()

    def _reset_counters(self):
        self.llm_calls = 0             # agent steps (each step = one LLM call)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.dropped_turns = 0
        self.ingest_errors = 0
        self.passages = 0
        self.turns = 0
        # What the agent spent its steps on (agent arm): archival inserts are
        # the only writes this bench can recall; core-memory edits live in the
        # prompt, not in a searchable store; replies are neither.
        self.archival_inserts = 0
        self.core_memory_edits = 0
        self.replies = 0
        self.other_tool_calls = 0

    def _c(self):
        if self._client is None:
            from letta_client import Letta  # imported late: only this adapter needs it
            self._client = Letta(base_url=URL, timeout=600.0)
        return self._client

    def open(self, run_dir: str) -> None:
        c = self._c()
        self._reset_counters()
        self.passage_item = {}
        self.known: set[str] = set()   # every passage id seen so far (agent arm)
        if self.server_version is None:
            try:
                self.server_version = c.health().version
            except Exception:  # noqa: BLE001
                self.server_version = "unknown"
        tag = hashlib.sha1(run_dir.encode("utf-8")).hexdigest()[:12]
        extra = {}
        if self.agent_driven:
            # ⚠ On server 0.16.8 a new agent's default type (letta_v1_agent) has
            # NO tools at all, and even memgpt_v2_agent's base set is only
            # conversation_search / memory_insert / memory_replace /
            # send_message — no archival tools. Without attaching them the
            # agent can never write a recallable memory and every turn reads
            # as "dropped" (our first pricing run measured exactly that — a
            # chatbot, not MemGPT). MemGPT's design is core memory + archival
            # memory the agent pages into, so both archival tools are attached.
            extra = {"agent_type": os.environ.get("BENCH_LETTA_AGENT_TYPE", "memgpt_v2_agent"),
                     "tools": ["archival_memory_insert", "archival_memory_search"]}
        a = c.agents.create(
            name=f"bench-{self.name}-{tag}",
            **llm_kwargs(),
            **extra,
            embedding_config=embedding_config(),
            memory_blocks=[{"label": "human", "value": ""},
                           {"label": "persona", "value": "I am a helpful assistant with long-term memory."}],
        )
        self.agent_id = a.id
        self.tools = sorted(t.name for t in (getattr(a, "tools", None) or []))
        if self.agent_driven and "archival_memory_insert" not in self.tools:
            raise RuntimeError(f"letta agent has no archival_memory_insert tool (tools={self.tools}); "
                               "it could never write a recallable memory")

    # -- ingest ---------------------------------------------------------
    def _when(self, it: MemoryItem):
        if it.when is None:
            return None
        try:
            return it.when.astimezone(timezone.utc).isoformat()
        except (AttributeError, ValueError):
            return None

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        for it in items:
            text = (it.text or "").strip()
            if not text:
                continue
            self.turns += 1
            if self.agent_driven:
                self._ingest_agent(it, text)
            else:
                self._ingest_archival(it, text)

    def _ingest_archival(self, it: MemoryItem, text: str) -> None:
        kw = {"text": text, "tags": [it.id]}
        w = self._when(it)
        if w:
            kw["created_at"] = w
        try:
            r = self._c().agents.passages.create(self.agent_id, **kw)
            self.passages += len(r) if isinstance(r, list) else 1
        except Exception as e:  # noqa: BLE001
            self.ingest_errors += 1
            if self.ingest_errors <= 3:
                print(f"[letta] insert failed for {it.id}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def _count_tools(self, messages) -> None:
        for m in messages:
            calls = getattr(m, "tool_calls", None) or []
            one = getattr(m, "tool_call", None)
            if one is not None:
                calls = [one, *calls]
            if getattr(m, "message_type", "") == "assistant_message" and not calls:
                self.replies += 1
            for tc in calls:
                name = (getattr(tc, "name", None) or "")
                if name == "archival_memory_insert":
                    self.archival_inserts += 1
                elif name.startswith("core_memory") or name.startswith("memory_") or name in ("memory", "rethink_memory"):
                    self.core_memory_edits += 1
                elif name == "send_message":
                    self.replies += 1
                else:
                    self.other_tool_calls += 1

    def _passage_ids(self) -> set[str]:
        out, after = set(), None
        while True:
            kw = {"limit": 1000, "ascending": True}
            if after:
                kw["after"] = after
            page = self._c().agents.passages.list(self.agent_id, **kw)
            page = list(page)
            out |= {p.id for p in page}
            if len(page) < 1000:
                return out
            after = page[-1].id

    def _ingest_agent(self, it: MemoryItem, text: str) -> None:
        kw = {"messages": [{"role": "user", "content": text[:4000]}]}
        if os.environ.get("BENCH_LETTA_MAX_STEPS"):
            kw["max_steps"] = int(os.environ["BENCH_LETTA_MAX_STEPS"])
        try:
            r = self._c().agents.messages.create(self.agent_id, **kw)
            self._count_tools(getattr(r, "messages", None) or [])
            u = getattr(r, "usage", None)
            if u is not None:
                self.llm_calls += int(getattr(u, "step_count", 0) or 0)
                self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)
        except Exception as e:  # noqa: BLE001
            self.ingest_errors += 1
            if self.ingest_errors <= 3:
                print(f"[letta] agent step failed for {it.id}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        # Whatever the agent archived while THIS turn was in flight came from it.
        now = self._passage_ids()
        new = now - self.known
        self.known = now
        for pid in new:
            self.passage_item[pid] = it.id
        self.passages += len(new)
        if not new:
            self.dropped_turns += 1

    # -- recall ---------------------------------------------------------
    def recall(self, query: str, k: int, when=None) -> list[RecallHit]:
        try:
            # Chunks of one turn share its tag: over-fetch so dedupe by item
            # still fills k (the same backfill the session cap uses).
            s = self._c().agents.passages.search(self.agent_id, query=query[:2000], top_k=max(k, 1) * 3)
        except Exception as e:  # noqa: BLE001
            print(f"[letta] search failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
            return []
        out, seen = [], set()
        results = getattr(s, "results", None) or []
        for rank, h in enumerate(results):
            tags = getattr(h, "tags", None) or []
            iid = tags[0] if tags and not self.agent_driven else self.passage_item.get(getattr(h, "id", ""))
            if not iid:
                # A passage with no attribution cannot be scored against a gold
                # turn; dropping it would shorten the list and flatter the row.
                iid = f"letta:{getattr(h, 'id', rank)}"
            if iid in seen:
                continue
            seen.add(iid)
            # The API returns no score; rank order is the ranking.
            out.append(RecallHit(id=iid, score=1.0 - rank / max(1, len(results)),
                                 text=str(getattr(h, "content", ""))[:200]))
            if len(out) == k:
                break
        return out

    # -- accounting -----------------------------------------------------
    def footprint_bytes(self) -> int:
        dsn = os.environ.get("BENCH_LETTA_PG_DSN")
        if not dsn or not self.agent_id:
            return 0
        try:
            import psycopg
            with psycopg.connect(dsn) as conn:
                r = conn.execute(
                    "SELECT coalesce(sum(pg_column_size(p.*)), 0) FROM archival_passages p "
                    "JOIN archives_agents aa ON aa.archive_id = p.archive_id WHERE aa.agent_id = %s",
                    (self.agent_id,)).fetchone()
                return int(r[0]) if r else 0
        except Exception as e:  # noqa: BLE001
            print(f"[letta] footprint query failed: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)
            return 0

    def ingest_stats(self) -> dict:
        """Numeric only (run.py sums these across stores)."""
        return {"llm_calls": self.llm_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "dropped_turns": self.dropped_turns,
                "ingest_errors": self.ingest_errors, "passages": self.passages, "turns": self.turns,
                "archival_inserts": self.archival_inserts, "core_memory_edits": self.core_memory_edits,
                "replies": self.replies, "other_tool_calls": self.other_tool_calls}

    def describe(self) -> dict:
        return {"server": "letta V1 API server (retired; last release 0.16.8)",
                "server_version": self.server_version, "arm": self.name,
                "llm": llm_kwargs() if self.agent_driven else None,
                "agent_type": os.environ.get("BENCH_LETTA_AGENT_TYPE", "memgpt_v2_agent") if self.agent_driven else "default",
                "tools": self.tools,
                "embedding": embedding_config()}

    def close(self) -> None:
        if self.agent_id and not os.environ.get("BENCH_LETTA_KEEP"):
            try:
                self._c().agents.delete(self.agent_id)
            except Exception:  # noqa: BLE001 — a failed delete must not lose the row
                pass
        self.agent_id = None


class LettaAgentAdapter(LettaArchivalAdapter):
    name = "letta_agent"
    agent_driven = True
