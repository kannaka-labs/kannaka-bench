"""kannaka — the binary, one HRM store per run.

`kannaka remember <text> --observed <iso>` per item (the id kannaka prints is
kept so hits map back to our item ids); `kannaka recall <query> --top-k k`
returns a JSON list. Swarm publishing is disabled by pointing the binary at
an unreachable NATS URL — a benchmark must never write into the production
swarm (it did, on the first probe, 2026-09-17). Shipped defaults otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import timezone
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit, dir_bytes

BIN = os.environ.get("KANNAKA_BIN", "kannaka")


class KannakaAdapter(Adapter):
    name = "kannaka"

    def __init__(self, bin_path: str = BIN, timeout_s: float = 60.0):
        self.bin = bin_path
        self.timeout_s = timeout_s
        self.dir = None
        self.env = None
        self.by_kid: dict[str, str] = {}      # kannaka memory id -> item id
        self.text_of: dict[str, str] = {}
        self.version = None

    def _run(self, args: list[str]) -> str:
        r = subprocess.run([self.bin] + args, env=self.env, capture_output=True, text=True,
                           timeout=self.timeout_s, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"kannaka {args[0]} failed: {r.stderr.strip()[:200]}")
        return r.stdout

    def open(self, run_dir: str) -> None:
        self.dir = os.path.join(run_dir, "kannaka")
        os.makedirs(self.dir, exist_ok=True)
        self.env = dict(os.environ, KANNAKA_DATA_DIR=self.dir,
                        KANNAKA_NATS_URL="nats://127.0.0.1:1",   # off the swarm, always
                        KANNAKA_FACET_DECOMPOSE=os.environ.get("KANNAKA_FACET_DECOMPOSE", "1"))
        try:
            self.version = subprocess.run([self.bin, "--version"], capture_output=True, text=True,
                                          timeout=10).stdout.strip()
        except Exception:
            self.version = "unknown"
        self.by_kid.clear()
        self.text_of.clear()

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        for it in items:
            args = ["remember", it.text[:4000]]
            if it.when is not None:
                iso = it.when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                args += ["--observed", iso, "--effective", iso]
            out = self._run(args)
            kid = None
            for line in reversed(out.strip().splitlines()):
                line = line.strip()
                if len(line) == 36 and line.count("-") == 4:
                    kid = line
                    break
            if kid:
                self.by_kid[kid] = it.id
            self.text_of[it.id] = it.text

    def recall(self, query: str, k: int, when=None) -> list[RecallHit]:
        out = self._run(["recall", query[:1000], "--top-k", str(k)])
        hits = []
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                try:
                    rows = json.loads(line)
                except ValueError:
                    rows = []
                for r in rows:
                    kid = r.get("id")
                    iid = self.by_kid.get(kid)
                    if iid is None:
                        # a memory kannaka made itself (a dream, a merge): keep it,
                        # it just cannot score
                        iid = f"kannaka:{kid}"
                    hits.append(RecallHit(id=iid, score=float(r.get("similarity") or 0.0),
                                          text=(r.get("content") or "")[:200]))
                break
        return hits[:k]

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir) if self.dir else 0

    def close(self) -> None:
        pass
