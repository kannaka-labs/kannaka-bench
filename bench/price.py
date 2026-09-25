"""Price an adapter's ingest BEFORE a full run.

Feeds the first N real turns of each chosen LongMemEval question through the
adapter, one turn at a time, and records per turn: wall seconds and the delta
in the adapter's own `ingest_stats()` (LLM calls, prompt/completion tokens,
drops, errors). Then extrapolates to the whole standard set (the sum of the
chosen questions' full haystacks), which is what a scored row would cost.

  python -m bench.price --adapter letta_agent --question-ids ~/std30-ids.txt \\
      --questions 2 --turns 8 --out ~/price-letta-agent.jsonl [--gpu-rate 0.87]

The turns are the dataset's own (so their length distribution is the real
one), taken from the START of each haystack — a fresh store, the cheapest
point of a run for systems whose per-turn cost grows with what they already
hold. The summary says so; a growth check needs a longer --turns.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time

from .run import make_adapter


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dataset", default="longmemeval_s")
    ap.add_argument("--question-ids", required=True, help="file, one question id per line (the standard set)")
    ap.add_argument("--questions", type=int, default=2, help="how many of those questions to sample turns from")
    ap.add_argument("--turns", type=int, default=8, help="turns per question")
    ap.add_argument("--out", required=True, help="per-turn jsonl")
    ap.add_argument("--gpu-rate", type=float, default=0.0,
                    help="$/h of the box the extrapolation should price (0 = time only)")
    a = ap.parse_args(argv)

    from .datasets import longmemeval
    qs, dsmeta = longmemeval.load(a.dataset, limit=None)
    keep = [l.strip() for l in open(a.question_ids, encoding="utf-8") if l.strip()]
    by_id = {q.id: q for q in qs}
    std = [by_id[i] for i in keep if i in by_id]
    total_turns = sum(len(q.items) for q in std)
    ad = make_adapter(a.adapter)
    rows = []
    work = tempfile.mkdtemp(prefix="bench-price-")
    with open(a.out, "w", encoding="utf-8") as f:
        for q in std[: a.questions]:
            ad.open(f"{work}/{q.id}")
            try:
                for it in q.items[: a.turns]:
                    before = dict(ad.ingest_stats())
                    t0 = time.perf_counter()
                    ad.ingest([it])
                    dt = time.perf_counter() - t0
                    after = ad.ingest_stats()
                    d = {k: after[k] - before.get(k, 0) for k in after if isinstance(after[k], (int, float))}
                    row = {"adapter": a.adapter, "question_id": q.id, "item_id": it.id,
                           "chars": len(it.text or ""), "seconds": round(dt, 3), **d}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(json.dumps(row), flush=True)
            finally:
                ad.close()
    if not rows:
        print("no turns measured")
        return 1
    secs = [r["seconds"] for r in rows]
    mean = statistics.mean(secs)
    summ = {
        "adapter": a.adapter, "turns_measured": len(rows),
        "chars_p50": statistics.median(r["chars"] for r in rows),
        "s_per_turn_p50": round(statistics.median(secs), 2), "s_per_turn_mean": round(mean, 2),
        "llm_calls_per_turn": round(sum(r.get("llm_calls", 0) for r in rows) / len(rows), 2),
        "prompt_tok_per_turn": round(sum(r.get("prompt_tokens", 0) for r in rows) / len(rows)),
        "completion_tok_per_turn": round(sum(r.get("completion_tokens", 0) for r in rows) / len(rows)),
        "errors": sum(r.get("ingest_errors", 0) for r in rows),
        "dropped": sum(r.get("dropped_turns", 0) for r in rows),
        "standard_set_turns": total_turns,
        "standard_set_hours_serial": round(total_turns * mean / 3600, 1),
        "note": "turns from the START of each haystack (fresh store); per-turn cost may grow with store size",
    }
    if a.gpu_rate:
        summ["standard_set_usd_serial"] = round(total_turns * mean / 3600 * a.gpu_rate, 2)
    print("SUMMARY " + json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
