"""Supermemory (github.com/supermemoryai/supermemory, MIT) through its self-hosted
server's HTTP API — the competitor row for issue #1.

  BENCH_SUPERMEMORY_URL   http://127.0.0.1:6767   (supermemory-server)
  BENCH_SUPERMEMORY_KEY   the sm_... key the server prints on first boot (Bearer)
  BENCH_SUPERMEMORY_MODE  documents | memories | hybrid   (search mode; default documents)
  BENCH_SUPERMEMORY_TASK  superrag | memory   (ingest task type; default superrag = chunk+embed
                          only, no LLM fact extraction — the retrieval-only comparison;
                          "memory" runs their extraction engine through whatever LLM the
                          server is configured with)

One store = one containerTag (derived from the run dir). Every item is one
document with customId = item id and metadata.item_id = item id, so a search
result maps back to the evidence unit the benchmark scores. Ingest waits for
the server's queue to drain (documents are processed asynchronously) by
polling the document status until nothing is left in a non-terminal state.
Two adapter names are exposed: `supermemory` (documents mode, superrag
ingest) and `supermemory_mem` (memories mode, memory ingest).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit

URL = os.environ.get("BENCH_SUPERMEMORY_URL", "http://127.0.0.1:6767").rstrip("/")
KEY = os.environ.get("BENCH_SUPERMEMORY_KEY", "")
TERMINAL = {"done", "completed", "failed", "error", "indexed", "ready"}


def _call(method: str, path: str, body=None, timeout: float = 120.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(URL + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


class SupermemoryAdapter(Adapter):
    name = "supermemory"
    search_mode = os.environ.get("BENCH_SUPERMEMORY_MODE", "documents")
    task_type = os.environ.get("BENCH_SUPERMEMORY_TASK", "superrag")

    def __init__(self):
        self.tag = ""
        self.doc_ids: list[str] = []
        self.by_doc: dict[str, str] = {}   # server document id -> item id
        self.last_timing_ms = None
        self.version = None

    def open(self, run_dir: str) -> None:
        h = hashlib.sha1(run_dir.encode("utf-8")).hexdigest()[:12]
        self.tag = f"bench_{h}"
        self.doc_ids = []
        self.by_doc = {}
        try:
            self.version = _call("GET", "/health", timeout=10)
        except Exception:
            self.version = None

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        for it in items:
            # customId allows only [A-Za-z0-9_:-]; "<session>#<turn>" carries '#'
            body = {"content": it.text, "containerTag": self.tag, "customId": it.id.replace("#", "__"),
                    "metadata": {"item_id": it.id, **{k: str(v) for k, v in (it.meta or {}).items() if isinstance(v, (str, int, float, bool))}},
                    "taskType": self.task_type}
            if it.when:
                body["documentDate"] = it.when.strftime("%Y-%m-%d")
            r = _call("POST", "/v3/documents", body)
            did = r.get("id")
            if did:
                self.doc_ids.append(did)
                self.by_doc[did] = it.id
        self._drain()

    def _drain(self, timeout_s: float = float(os.environ.get("BENCH_SUPERMEMORY_DRAIN_S", "7200"))):
        """Wait until every document of this store reports a terminal status
        (GET /v3/documents/{id}: queued -> extracting -> embedding -> indexing
        -> done). A store searched before its documents are indexed would
        score retrieval against a partial index and look like a miss, so an
        incomplete drain is an ERROR for the row, never a silent partial.
        (2026-09-17: the server paused its ingest queue at its 1 GB ingest
        memory limit and the old give-up-after-60 s drain let two questions
        through against half-built stores.)"""
        t0 = time.time()
        pending = list(self.doc_ids)
        while pending and time.time() - t0 < timeout_s:
            still = []
            for did in pending:
                try:
                    st = str(_call("GET", f"/v3/documents/{did}", timeout=30).get("status") or "").lower()
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        continue
                    st = "unknown"
                except Exception:
                    st = "unknown"
                if st in TERMINAL:
                    continue
                still.append(did)
            pending = still
            if pending:
                time.sleep(5)
        if pending:
            raise RuntimeError(f"supermemory ingest incomplete after {int(time.time() - t0)}s: {len(pending)} of "
                               f"{len(self.doc_ids)} documents not done (server queue paused or stalled?)")

    def recall(self, query: str, k: int, when: datetime | None = None) -> list[RecallHit]:
        body = {"q": query[:2000], "containerTag": self.tag, "limit": min(max(k, 1), 100),
                "searchMode": self.search_mode, "threshold": 0.0}
        r = _call("POST", "/v4/search", body)
        self.last_timing_ms = r.get("timing")
        out: list[RecallHit] = []
        seen = set()
        for res in r.get("results") or []:
            item_id = None
            md = res.get("metadata") or {}
            if isinstance(md, dict):
                item_id = md.get("item_id")
            if not item_id:
                cid = res.get("customId")
                item_id = (cid.replace("__", "#") if cid else None) or self.by_doc.get(res.get("documentId") or res.get("id") or "")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            out.append(RecallHit(id=item_id, score=float(res.get("similarity") or res.get("score") or 0.0),
                                 text=res.get("memory") or res.get("chunk") or res.get("content") or ""))
            if len(out) >= k:
                break
        return out

    def footprint_bytes(self) -> int:
        return 0   # the server's store is shared; per-store bytes are not exposed

    def close(self) -> None:
        """Delete this store's documents. "Self-hosted lite is licensed for up
        to 10,000 documents" (HTTP 403 document_limit_reached, 2026-09-17,
        after 19 LongMemEval stores of ~500 turns); the run keeps its rows, the
        server only ever holds one store. BENCH_SUPERMEMORY_KEEP=1 keeps them."""
        if os.environ.get("BENCH_SUPERMEMORY_KEEP") == "1":
            return
        for did in self.doc_ids:
            try:
                _call("DELETE", f"/v3/documents/{did}", timeout=30)
            except Exception:
                pass
        self.doc_ids = []


class SupermemoryMemAdapter(SupermemoryAdapter):
    """Their memory engine: LLM fact extraction on ingest, `memories` search."""
    name = "supermemory_mem"
    search_mode = "memories"
    task_type = "memory"
