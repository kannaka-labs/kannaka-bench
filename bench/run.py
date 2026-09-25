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
import shutil
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
    "mem0": "bench.adapters.mem0:Mem0Adapter",
    "pgvector": "bench.adapters.pgvector:PgvectorAdapter",
    "pgvector_exact": "bench.adapters.pgvector:PgvectorExactAdapter",
    "letta_archival": "bench.adapters.letta:LettaArchivalAdapter",
    "letta_agent": "bench.adapters.letta:LettaAgentAdapter",
    "graphiti": "bench.adapters.graphiti:GraphitiAdapter",
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


def recall_flags() -> dict:
    """The kannaka ranking knobs in force for this process.

    Returns `{}` when none are set — and that empty dict is a result, not a
    missing value. The caller must record it either way: an arm that ran with
    the flag off is the control, and a control indistinguishable from a
    recording failure is not a control.
    """
    return {k: v for k, v in os.environ.items()
            if k.startswith("KANNAKA_RECALL_") or k in ("KANNAKA_ENCODER",)}


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


def parse_shard(spec: str | None) -> tuple[int, int]:
    """'I/N' -> (I, N), 0 <= I < N. None -> (0, 1): every store."""
    if not spec:
        return 0, 1
    i, _, n = spec.partition("/")
    i, n = int(i), int(n)
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard wants I/N with 0 <= I < N, got {spec!r}")
    return i, n


def shard_stores(stores: list, spec: str | None) -> list:
    """Store-level round-robin split: shard I of N keeps stores I, I+N, I+2N...
    in the loader's order, so N shards together run every store exactly once
    and each shard gets a mix of question types (the loader is in file order,
    which is grouped by type)."""
    i, n = parse_shard(spec)
    return [st for j, st in enumerate(stores) if j % n == i]


def run_store(adapter: Adapter, run_dir: str, items, questions, k: int, level: str, rows: list, ds: str,
              cap: int = 0, consolidate: bool = False):
    """One store: ingest all items once, then ask every question against it.
    With a session cap the adapter is asked for k*3 candidates and the cap
    picks the k that go into the row (the row records both)."""
    k_fetch = k * 3 if cap > 0 else k
    adapter.open(run_dir)
    t0 = time.perf_counter()
    adapter.ingest(items)
    cons = adapter.consolidate() if consolidate else {}
    ingest_ms = (time.perf_counter() - t0) * 1000.0
    per_item = ingest_ms / max(1, len(items))
    footprint = adapter.footprint_bytes()
    # Many questions on one store (LoCoMo): one process when the adapter can.
    if len(questions) > 1 and hasattr(adapter, "recall_many"):
        t0 = time.perf_counter()
        # Dates go with the questions: the kannaka adapter scores temporal
        # recency as of when each was asked, which on a dated corpus is the
        # difference between the temporal factor ranking and being a constant.
        all_hits = adapter.recall_many(
            [q.question for q in questions], k_fetch, [q.asked_at for q in questions]
        )
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
            "consolidate": cons, "dream_hits_at_k": sum(1 for h in ids[:k] if h.startswith("kannaka:")),
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
    ap.add_argument("--consolidate", action="store_true",
                    help="run each adapter's consolidation step (kannaka: a dream cycle) after ingest, before recall")
    ap.add_argument("--max-items", type=int, default=0,
                    help="stores with more items than this are skipped (error row) for --max-items-adapters; 0 = no cap")
    ap.add_argument("--max-items-adapters", default="kannaka,kannaka_minilm",
                    help="comma list of adapters the --max-items cap applies to")
    ap.add_argument("--question-ids", default=None,
                    help="file with one question_id per line: run exactly these questions (E-L3c's held-out set)")
    ap.add_argument("--shard", default=None,
                    help="I/N: run only stores I, I+N, ... (merge shard dirs with `python -m bench.merge`)")
    ap.add_argument("--drop-stores", action="store_true",
                    help="delete each store's directory once it is scored (LongMemEval-M: ~0.5 GB per kannaka store)")
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
        keep = None
        if a.question_ids:
            keep = {l.strip() for l in open(a.question_ids, encoding="utf-8") if l.strip()}
        qs, dsmeta = longmemeval.load(a.dataset, limit=a.limit, question_ids=keep)
        if keep is not None:
            print(f"[run] --question-ids: {len(qs)} of {len(keep)} requested questions found", flush=True)
        stores = [(q.id, q.items, [q], "session") for q in qs]
    stores_total = len(stores)
    stores = shard_stores(stores, a.shard)

    adapters = {n: make_adapter(n) for n in names}
    manifest = {
        "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(), "commit": git_commit(),
        "dataset": dsmeta, "adapters": names, "k": a.k, "session_cap": a.session_cap, "limit": a.limit,
        "max_items": a.max_items, "consolidate": a.consolidate,
        "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(), "stores": len(stores),
        "shard": a.shard, "stores_total": stores_total,
        "adapter_versions": {},
        # Ranking flags belong beside the numbers they produced. Three arms of a
        # temporal A/B were written with no record of which flag each ran with,
        # and the store that would have shown it is deleted at the end of the
        # run — so the arms were only distinguishable by trusting shell history.
        #
        # Recorded HERE, once, rather than per-adapter after a successful
        # run_store, for two reasons. The environment is process-global and
        # cannot differ between adapters, so the per-adapter capture was
        # measuring the same thing N times; and an adapter that raised left no
        # entry, so a missing key meant either "no flags" or "this arm died".
        # An EMPTY dict is the finding in the control arm — the arm with the
        # flag off is exactly the one the old `if flags:` guard recorded
        # nothing for, which defeated the fix in the case it was written for.
        "kannaka_flags": {n: dict(recall_flags()) for n in names},
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
        # A start line, not only the completion line below. The completion line
        # prints after EVERY adapter has finished the store, so a slow ingest
        # produced no output at all: a mem0 pilot ran 39 minutes in silence and
        # "working" was indistinguishable from "hung" — the log had to be
        # diagnosed from the vector store's byte count instead.
        print(f"[{si + 1}/{len(stores)}] {sid}: START {len(items)} items, "
              f"{len(questions)} q, adapters {','.join(adapters)}", flush=True)
        for n, ad in adapters.items():
            t_ad = time.perf_counter()
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
                run_store(ad, run_dir, items, questions, a.k, level, rows, a.dataset, cap=a.session_cap,
                          consolidate=a.consolidate)
                # Every kannaka arm, not only the one named "kannaka": the
                # standard-setting manifest (kannaka_minilm) recorded no binary
                # at all, so which build produced the published row lived only
                # in prose. Version string + resolved path + sha256.
                if getattr(ad, "version", None):
                    manifest["adapter_versions"][n] = ad.version
                if getattr(ad, "bin_info", None):
                    manifest.setdefault("adapter_bins", {})[n] = ad.bin_info
                # The configuration a competitor row ran with (index, versions,
                # extraction model) — part of what is under test.
                if hasattr(ad, "describe"):
                    manifest.setdefault("adapter_config", {})[n] = ad.describe()
                # Whether the attention beam actually fired, and how sparse it
                # was. Without this a "beam" arm and a dense arm are
                # indistinguishable in the record.
                if hasattr(ad, "beam_stats"):
                    manifest.setdefault("beam", {})[n] = ad.beam_stats()
                # Mem0 spends an LLM call per ingested turn and can drop turns
                # in extraction; every other adapter ingests free and lossless.
                # Without these columns its row reads as like-for-like.
                if hasattr(ad, "ingest_stats"):
                    manifest.setdefault("ingest_stats", {})[n] = ad.ingest_stats()
                print(f"[{si + 1}/{len(stores)}] {sid}: {n} done in "
                      f"{time.perf_counter() - t_ad:.1f}s", flush=True)
            except Exception as e:
                for q in questions:
                    rows.append({"dataset": a.dataset, "adapter": n, "question_id": q.id, "qtype": q.qtype,
                                 "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"[{sid}] {n}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            if a.drop_stores:
                shutil.rmtree(os.path.join(work, sid), ignore_errors=True)
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
