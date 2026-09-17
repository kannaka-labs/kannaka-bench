"""LongMemEval (Wu et al., 2024): each question carries its own haystack of
chat sessions; the evidence is a set of session ids. We ingest one item per
turn, id "<session_id>#<turn_index>", timestamped with the session date, and
score a hit when a retrieved item's session is in the gold set.

Files: huggingface.co/datasets/xiaowu0162/longmemeval — longmemeval_oracle
(evidence sessions only), longmemeval_s, longmemeval_m. Downloaded once to
~/.kannaka-bench/data and named by sha256 in every run manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..adapters.base import MemoryItem

DATA_DIR = os.path.expanduser(os.environ.get("KANNAKA_BENCH_DATA", "~/.kannaka-bench/data"))
URLS = {
    "longmemeval_oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle",
    "longmemeval_s": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s",
    "longmemeval_m": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_m",
}


@dataclass
class Question:
    id: str
    question: str
    answer: str
    qtype: str
    asked_at: datetime | None
    items: list[MemoryItem]          # the haystack, in history order
    gold_ids: set[str]               # ids that count as a hit (session ids or turn ids)
    gold_level: str = "session"      # "session": hit if item.id.split('#')[0] in gold_ids
    meta: dict = field(default_factory=dict)


def parse_date(s: str) -> datetime | None:
    """'2023/04/10 (Mon) 23:07' -> aware UTC datetime."""
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{1,2}):(\d{2})", s or "")
    if not m:
        return None
    y, mo, d, h, mi = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(variant: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, variant + ".json")
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(URLS[variant], tmp)
        os.replace(tmp, path)
    return path


def stratified(data: list[dict], limit: int | None) -> list[dict]:
    """The first N questions of the file are not a sample of it (the first 50
    of longmemeval_s are all single-session-user). Take N per question type,
    in file order within a type, then restore file order."""
    if not limit:
        return list(data)
    taken: dict[str, int] = {}
    keep = []
    for i, q in enumerate(data):
        t = q.get("question_type", "")
        if taken.get(t, 0) < limit:
            taken[t] = taken.get(t, 0) + 1
            keep.append(i)
    return [data[i] for i in keep]


def questions_from(data: list[dict], limit: int | None = None) -> list[Question]:
    out = []
    for q in stratified(data, limit):
        items: list[MemoryItem] = []
        sids = q.get("haystack_session_ids") or []
        dates = q.get("haystack_dates") or []
        sessions = q.get("haystack_sessions") or []
        # history order = by date, stable; sessions arrive shuffled in the file
        order = sorted(range(len(sessions)), key=lambda i: (parse_date(dates[i]) if i < len(dates) and dates[i] else datetime.min.replace(tzinfo=timezone.utc)))
        for i in order:
            sid = sids[i] if i < len(sids) else f"s{i}"
            when = parse_date(dates[i]) if i < len(dates) else None
            for t, turn in enumerate(sessions[i]):
                role = turn.get("role", "user")
                text = (turn.get("content") or "").strip()
                if not text:
                    continue
                items.append(MemoryItem(id=f"{sid}#{t}", text=f"{role}: {text}", when=when,
                                        meta={"session": sid, "has_answer": bool(turn.get("has_answer"))}))
        gold = set(q.get("answer_session_ids") or [])
        if not gold:
            gold = {sids[i] for i, s in enumerate(sessions) if any(t.get("has_answer") for t in s) and i < len(sids)}
        out.append(Question(id=q["question_id"], question=q["question"], answer=str(q.get("answer", "")),
                            qtype=q.get("question_type", ""), asked_at=parse_date(q.get("question_date", "")),
                            items=items, gold_ids=gold, gold_level="session"))
    return out


def load(variant: str = "longmemeval_oracle", limit: int | None = None) -> tuple[list[Question], dict]:
    path = fetch(variant)
    data = json.load(open(path, encoding="utf-8"))
    return questions_from(data, limit), {"dataset": variant, "path": path, "sha256": sha256_file(path), "questions_total": len(data),
                                         "limit_is_per_type": bool(limit)}
