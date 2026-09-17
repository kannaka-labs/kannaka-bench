"""Run a benchmark: every adapter, same items, same order, same k.

  python -m bench.run --dataset longmemeval_oracle --adapters kannaka,vector_numpy,recency --k 5 --limit 20 --out results/

Writes <out>/<run-id>/manifest.json and results.jsonl (one row per question
per adapter): hits, any_hit@k, recall@k, mrr, recall_ms, ingest_ms per item,
footprint bytes. Phase 1 = retrieval only; the answer model comes in phase 2.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from . import metrics
from .adapters.base import Adapter

ADAPTERS = {
    "kannaka": "bench.adapters.kannaka:KannakaAdapter",
    "vector_numpy": "bench.adapters.baselines:VectorNumpyAdapter",
    "recency": "bench.adapters.baselines:RecencyAdapter",
}


def make_adapter(name: str) -> Adapter:
    mod, cls = ADAPTERS[name].split(":")
    m = __import__(mod, fromlist=[cls])
    return getattr(m, cls)()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
    except Exception:
        return "unknown"


def run_store(adapter: Adapter, run_dir: str, items, questions, k: int, level: str, rows: list, ds: str):
    """One store: ingest all items once, then ask every question against it."""
    adapter.open(run_dir)
    t0 = time.perf_counter()
    adapter.ingest(items)
    ingest_ms = (time.perf_counter() - t0) * 1000.0
    per_item = ingest_ms / max(1, len(items))
    footprint = adapter.footprint_bytes()
    # Many questions on one store (LoCoMo): one process when the adapter can.
    if len(questions) > 1 and hasattr(adapter, "recall_many"):
        t0 = time.perf_counter()
        all_hits = adapter.recall_many([q.question for q in questions], k)
        per_q_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(questions))
        timed_hits = [(h, per_q_ms) for h in all_hits]
    else:
        timed_hits = [Adapter.timed(adapter.recall, q.question, k, q.asked_at) for q in questions]
    for q, (hits, ms) in zip(questions, timed_hits):
        ids = [h.id for h in hits]
        rows.append({
            "dataset": ds, "adapter": adapter.name, "question_id": q.id, "qtype": q.qtype,
            "k": k, "n_items": len(items), "gold": sorted(q.gold_ids), "gold_level": level,
            "hits": ids, "any_hit_at_k": metrics.any_hit_at_k(ids, q.gold_ids, level, k),
            "recall_at_k": metrics.recall_at_k(ids, q.gold_ids, level, k),
            "mrr": metrics.mrr(ids, q.gold_ids, level),
            "recall_ms": round(ms, 2), "ingest_ms_per_item": round(per_item, 3),
            "footprint_bytes": footprint,
        })
    adapter.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="longmemeval_oracle",
                    choices=["longmemeval_oracle", "longmemeval_s", "longmemeval_m", "locomo"])
    ap.add_argument("--adapters", default="kannaka,vector_numpy,recency")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="questions (LongMemEval) / questions per conversation (LoCoMo)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)

    names = [n for n in a.adapters.split(",") if n]
    run_id = a.name or f"{a.dataset}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = os.path.join(a.out, run_id)
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="kannaka-bench-")

    if a.dataset == "locomo":
        from .datasets import locomo
        convs, dsmeta = locomo.load(limit_questions=a.limit)
        stores = [(cid, items, qs, "turn") for cid, items, qs in convs]
    else:
        from .datasets import longmemeval
        qs, dsmeta = longmemeval.load(a.dataset, limit=a.limit)
        stores = [(q.id, q.items, [q], "session") for q in qs]

    adapters = {n: make_adapter(n) for n in names}
    manifest = {
        "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(), "commit": git_commit(),
        "dataset": dsmeta, "adapters": names, "k": a.k, "limit": a.limit,
        "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(), "stores": len(stores),
        "adapter_versions": {},
    }
    rows: list[dict] = []
    t_start = time.perf_counter()
    for si, (sid, items, questions, level) in enumerate(stores):
        for n, ad in adapters.items():
            run_dir = os.path.join(work, sid)
            try:
                run_store(ad, run_dir, items, questions, a.k, level, rows, a.dataset)
                if n == "kannaka" and getattr(ad, "version", None):
                    manifest["adapter_versions"]["kannaka"] = ad.version
            except Exception as e:
                for q in questions:
                    rows.append({"dataset": a.dataset, "adapter": n, "question_id": q.id, "qtype": q.qtype,
                                 "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"[{sid}] {n}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        done = sum(1 for r in rows if "error" not in r)
        print(f"[{si + 1}/{len(stores)}] {sid}: {len(items)} items, {len(questions)} q, rows so far {done}", flush=True)
        with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_s"] = round(time.perf_counter() - t_start, 1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out_dir}/results.jsonl ({len(rows)} rows) and manifest.json")
    from . import report
    print(report.render(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
