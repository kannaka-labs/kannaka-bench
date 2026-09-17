"""Retrieval metrics. A hit is judged at the dataset's evidence level:
session (LongMemEval: item id "<session>#<turn>") or turn (LoCoMo: "D1:3")."""
from __future__ import annotations


def evidence_key(item_id: str, level: str) -> str:
    return item_id.split("#", 1)[0] if level == "session" else item_id


def hit_ranks(hit_ids: list[str], gold_ids: set[str], level: str) -> list[int]:
    """1-based ranks of retrieved items that are gold, in order."""
    return [i + 1 for i, h in enumerate(hit_ids) if evidence_key(h, level) in gold_ids]


def recall_at_k(hit_ids: list[str], gold_ids: set[str], level: str, k: int) -> float:
    """Fraction of gold evidence units covered by the top-k (1.0 when all are)."""
    if not gold_ids:
        return 0.0
    seen = {evidence_key(h, level) for h in hit_ids[:k]}
    return len(seen & gold_ids) / len(gold_ids)


def any_hit_at_k(hit_ids: list[str], gold_ids: set[str], level: str, k: int) -> bool:
    return any(evidence_key(h, level) in gold_ids for h in hit_ids[:k])


def mrr(hit_ids: list[str], gold_ids: set[str], level: str) -> float:
    r = hit_ranks(hit_ids, gold_ids, level)
    return 1.0 / r[0] if r else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)
