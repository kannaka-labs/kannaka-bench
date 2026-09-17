"""LoCoMo (Maharana et al., 2024): ten very long two-person conversations,
each with ~200 questions whose evidence is a set of dialogue turns ("D1:3").
One store per conversation; one item per turn, id = the dataset's dia_id,
timestamped with the session's date; a hit is a retrieved gold turn.

File: github.com/snap-research/locomo data/locomo10.json.
Categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone

from ..adapters.base import MemoryItem
from .longmemeval import DATA_DIR, Question, sha256_file

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def parse_dt(s: str) -> datetime | None:
    """'1:56 pm on 8 May, 2023' -> aware UTC datetime."""
    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})", (s or "").strip(), re.I)
    if not m:
        return None
    h, mi, ap, d, mon, y = m.groups()
    h = int(h) % 12 + (12 if ap.lower() == "pm" else 0)
    try:
        month = datetime.strptime(mon[:3], "%b").month
    except ValueError:
        return None
    return datetime(int(y), month, int(d), h, int(mi), tzinfo=timezone.utc)


def fetch() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "locomo10.json")
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, path)
    return path


def conversations_from(data: list[dict], limit_questions: int | None = None) -> list[tuple[str, list[MemoryItem], list[Question]]]:
    """[(conversation_id, items, questions)]. Questions in a conversation share its store."""
    out = []
    for ci, conv in enumerate(data):
        c = conv.get("conversation") or {}
        items: list[MemoryItem] = []
        n = 1
        while f"session_{n}" in c:
            when = parse_dt(c.get(f"session_{n}_date_time", ""))
            for turn in c[f"session_{n}"]:
                text = (turn.get("text") or "").strip()
                did = turn.get("dia_id") or f"D{n}:{len(items)}"
                if not text:
                    continue
                if turn.get("blip_caption"):
                    text += f" [shared an image: {turn['blip_caption']}]"
                items.append(MemoryItem(id=did, text=f"{turn.get('speaker', '?')}: {text}", when=when,
                                        meta={"session": f"D{n}"}))
            n += 1
        qs = []
        for qi, q in enumerate(conv.get("qa") or []):
            gold = set(q.get("evidence") or [])
            cat = q.get("category")
            qs.append(Question(id=f"c{ci}-q{qi}", question=q.get("question", ""), answer=str(q.get("answer", "")),
                               qtype=CATEGORY.get(cat, str(cat)), asked_at=None, items=[], gold_ids=gold,
                               gold_level="turn", meta={"conversation": ci, "category": cat}))
        if limit_questions:
            qs = qs[:limit_questions]
        out.append((f"c{ci}", items, qs))
    return out


def load(limit_questions: int | None = None) -> tuple[list, dict]:
    path = fetch()
    data = json.load(open(path, encoding="utf-8"))
    return conversations_from(data, limit_questions), {"dataset": "locomo10", "path": path, "sha256": sha256_file(path), "conversations": len(data)}
