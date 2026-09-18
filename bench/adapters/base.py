"""One interface for every memory system under test.

An adapter is opened once per store (LongMemEval: one per question, because
each question carries its own haystack; LoCoMo: one per conversation), fed
items in history order, asked for the top-k, and closed. It reports its own
footprint. It never sees the questions before ingest is done, and never sees
the gold ids.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable


@dataclass
class MemoryItem:
    id: str                      # "<session_id>#<turn>" or "D1:3" — the unit of evidence
    text: str
    when: datetime | None = None  # when it was said, if the dataset knows
    meta: dict = field(default_factory=dict)


@dataclass
class RecallHit:
    id: str
    score: float
    text: str = ""


class Adapter:
    name = "base"

    def open(self, run_dir: str) -> None:
        """Create an empty store under run_dir."""

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        raise NotImplementedError

    def consolidate(self) -> dict:
        """Optional: run the system's offline consolidation after ingest and
        before any recall (kannaka: a dream cycle). Returns what it did, for
        the row; {} when the system has no such step."""
        return {}

    def recall(self, query: str, k: int, when: datetime | None = None) -> list[RecallHit]:
        raise NotImplementedError

    def footprint_bytes(self) -> int:
        return 0

    def close(self) -> None:
        """Release resources; the store on disk may stay for inspection."""

    # -- helpers shared by adapters -------------------------------------

    @staticmethod
    def timed(fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        return out, (time.perf_counter() - t0) * 1000.0


def dir_bytes(path: str) -> int:
    import os
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total
