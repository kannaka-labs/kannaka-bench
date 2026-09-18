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
    "kannaka_minilm": "bench.adapters.kannaka:KannakaMinilmAdapter",
    "vector_numpy": "bench.adapters.baselines:VectorNumpyAdapter",
    "recency": "bench.adapters.baselines:RecencyAdapter",
    "kannaka_bge": "bench.adapters.kannaka:KannakaBgeAdapter",
    "vector_bge": "bench.adapters.baselines:VectorBgeAdapter",
    "supermemory": "bench.adapters.supermemory:SupermemoryAdapter",
    "supermemory_mem": "bench.adapters.supermemory:SupermemoryMemAdapter",
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


def session_cap(hits, k: int, cap: int, level: str):
    """Keep at most `cap` hits per session (item id "<session>#<turn>"), in rank
    order, then cut to k. With cap=0 or a turn-level dataset, plain top-k. The
    candidate list is deeper than k so the cut is backfilled, not just thinned:
    a session that owned four of the top-15 slots gives three of them to the
    next distinct sessions."""
    if cap <= 0 or level != "session":
        return hits[:k]
    seen: dict = {}
    out = []
    for h in hits:
        sid = h.id.rpartition("#")[0] if "#" in h.id else h.id
        if seen.get(sid, 0) >= cap:
            continue
        seen[sid] = seen.get(sid, 0) + 1
        out.append(h)
        if len(out) == k:
            break
    return out


def run_store(adapter: Adapter, run_dir: str, items, questions, k: int, level: str, rows: list, ds: str,
              cap: int = 0):
    """One store: ingest all items once, then ask every question against it.
    With a session cap the adapter is asked for k*3 candidates and the cap
    picks the k that go into the row (the row records both)."""
    k_fetch = k * 3 if cap > 0 else k
    adapter.open(run_dir)
    t0 = time.perf_counter()
    adapter.ingest(items)
    ingest_ms = (time.perf_counter() - t0) * 1000.0
    per_item = ingest_ms / max(1, len(items))
    footprint = adapter.footprint_bytes()
    # Many questions on one store (LoCoMo): one process when the adapter can.
    if len(questions) > 1 and hasattr(adapter, "recall_many"):
        t0 = time.perf_counter()
        all_hits = adapter.recall_many([q.question for q in questions], k_fetch)
        per_q_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(questions))
        timed_hits = [(h, per_q_ms) for h in all_hits]
    else:
        timed_hits = [Adapter.timed(adapter.recall, q.question, k_fetch, q.asked_at) for q in questions]
    evidence_all = {it.id for it in items if (it.meta or {}).get("has_answer")}
    for q, (hits, ms) in zip(questions, timed_hits):
        candidates = [h.id for h in hits]          # everything the adapter returned, rank order
        hits = session_cap(hits, k, cap, level)
        ids = [h.id for h in hits]
        # answer-bearing turns of THIS question's gold sessions (LongMemEval marks
        # has_answer per turn; ConvoMem's evidence messages are matched by text)
        evidence = {e for e in evidence_all if (e.rpartition("#")[0] if "#" in e else e) in q.gold_ids} if q.gold_ids else set()
        ev_cov = (len(set(ids[:k]) & evidence) / len(evidence)) if evidence else None
        rows.append({
            "dataset": ds, "adapter": adapter.name, "question_id": q.id, "qtype": q.qtype,
            "k": k, "session_cap": cap, "k_fetch": k_fetch, "n_items": len(items), "gold": sorted(q.gold_ids), "gold_level": level,
            "candidates": candidates,
            "hits": ids, "any_hit_at_k": metrics.any_hit_at_k(ids, q.gold_ids, level, k),
            "evidence_turns": len(evidence), "evidence_coverage_at_k": ev_cov,
            "recall_at_k": metrics.recall_at_k(ids, q.gold_ids, level, k),
            "mrr": metrics.mrr(ids, q.gold_ids, level),
            "recall_ms": round(ms, 2), "ingest_ms_per_item": round(per_item, 3),
            "footprint_bytes": footprint,
        })
    adapter.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="longmemeval_oracle",
                    choices=["longmemeval_oracle", "longmemeval_s", "longmemeval_m", "locomo", "convomem"])
    ap.add_argument("--convomem-context", type=int, default=50,
                    help="convomem: haystack size in conversations (1..300 as pre-mixed on the hub)")
    ap.add_argument("--adapters", default="kannaka,vector_numpy,recency")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--session-cap", type=int, default=0,
                    help="at most N turns per session in the top-k (0 = off); candidates fetched at k*3")
    ap.add_argument("--limit", type=int, default=None,
                    help="LongMemEval: questions PER question type (stratified); LoCoMo: questions per conversation")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-items", type=int, default=0,
                    help="stores with more items than this are skipped (error row) for --max-items-adapters; 0 = no cap")
    ap.add_argument("--max-items-adapters", default="kannaka,kannaka_minilm",
                    help="comma list of adapters the --max-items cap applies to")
    ap.add_argument("--resume", action="store_true",
                    help="keep the run dir's existing non-error rows and skip those (adapter, question) pairs")
    a = ap.parse_args(argv)

    names = [n for n in a.adapters.split(",") if n]
    run_id = a.name or f"{a.dataset}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = os.path.join(a.out, run_id)
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="kannaka-bench-")

    if a.dataset == "convomem":
        from .datasets import convomem
        qs, dsmeta = convomem.load(context=a.convomem_context, limit=a.limit)
        stores = [(q.id.replace("/", "_"), q.items, [q], "session") for q in qs]
    elif a.dataset == "locomo":
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
        "dataset": dsmeta, "adapters": names, "k": a.k, "session_cap": a.session_cap, "limit": a.limit,
        "max_items": a.max_items,
        "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(), "stores": len(stores),
        "adapter_versions": {},
    }
    rows: list[dict] = []
    finished: set = set()      # (adapter, question id) pairs kept from a --resume
    prev = os.path.join(out_dir, "results.jsonl")
    if a.resume and os.path.exists(prev):
        for line in open(prev, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    rows.append(r)
                    finished.add((r["adapter"], r["question_id"]))
        print(f"resume: keeping {len(rows)} rows, {len(finished)} (adapter, question) pairs done", flush=True)
    t_start = time.perf_counter()
    for si, (sid, items, questions, level) in enumerate(stores):
        for n, ad in adapters.items():
            if finished and all((n, q.id) in finished for q in questions):
                continue
            if a.max_items and len(items) > a.max_items and n in set(a.max_items_adapters.split(",")):
                for q in questions:
                    rows.append({"dataset": a.dataset, "adapter": n, "question_id": q.id, "qtype": q.qtype,
                                 "n_items": len(items),
                                 "error": f"skipped: n_items {len(items)} > max_items {a.max_items} (kannaka-memory #978)"})
                print(f"[{sid}] {n}: SKIPPED {len(items)} items > --max-items {a.max_items}", flush=True)
                continue
            run_dir = os.path.join(work, sid)
            try:
                run_store(ad, run_dir, items, questions, a.k, level, rows, a.dataset, cap=a.session_cap)
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
