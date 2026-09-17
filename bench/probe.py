"""Ablation probe for the kannaka adapter: the same questions under different
kannaka settings, so a loss can be explained before it is published.

  python -m bench.probe --dataset longmemeval_s --limit 1 --k 5 \
      --configs default,nofacet,notime,nofacet-notime --out results/probe/

Each config is an environment for the kannaka binary plus a switch on the
adapter (timestamps on/off). Prints hit@k per config per question and the
per-config totals. Retrieval only, no answer model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from . import metrics
from .adapters.base import Adapter, MemoryItem
from .adapters.kannaka import KannakaAdapter, KannakaMinilmAdapter

ADAPTERS = {"kannaka": KannakaAdapter, "kannaka_minilm": KannakaMinilmAdapter}

CONFIGS = {
    "default": {"env": {}, "timestamps": True},
    "nofacet": {"env": {"KANNAKA_FACET_DECOMPOSE": "0"}, "timestamps": True},
    "notime": {"env": {}, "timestamps": False},
    "nofacet-notime": {"env": {"KANNAKA_FACET_DECOMPOSE": "0"}, "timestamps": False},
    # the xi-diversity reranker (kannaka-memory #975): rank by raw resonance instead
    "xioff": {"env": {"KANNAKA_RECALL_XI_BOOST": "off"}, "timestamps": True},
    "xioff-nofacet": {"env": {"KANNAKA_RECALL_XI_BOOST": "off", "KANNAKA_FACET_DECOMPOSE": "0"}, "timestamps": True},
}


def strip_time(items):
    return [MemoryItem(id=it.id, text=it.text, when=None, meta=it.meta) for it in items]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="longmemeval_s")
    ap.add_argument("--limit", type=int, default=1, help="questions per type")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--configs", default="default,nofacet,notime,nofacet-notime")
    ap.add_argument("--adapter", default="kannaka", choices=sorted(ADAPTERS))
    ap.add_argument("--out", default="results/probe")
    a = ap.parse_args(argv)

    from .datasets import longmemeval
    qs, dsmeta = longmemeval.load(a.dataset, limit=a.limit)
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="kannaka-probe-")
    base_env = dict(os.environ)
    rows = []
    totals = {}
    for cname in [c for c in a.configs.split(",") if c]:
        cfg = CONFIGS[cname]
        os.environ.clear()
        os.environ.update(base_env)
        os.environ.update(cfg["env"])
        hits_n = 0
        for q in qs:
            ad = ADAPTERS[a.adapter]()
            run_dir = os.path.join(work, cname, q.id)
            os.makedirs(run_dir, exist_ok=True)
            ad.open(run_dir)
            items = q.items if cfg["timestamps"] else strip_time(q.items)
            t0 = time.perf_counter()
            ad.ingest(items)
            ingest_s = time.perf_counter() - t0
            hits, ms = Adapter.timed(ad.recall, q.question, a.k, q.asked_at)
            ids = [h.id for h in hits]
            hit = metrics.any_hit_at_k(ids, q.gold_ids, "session", a.k)
            hits_n += int(hit)
            row = {"config": cname, "adapter": a.adapter, "question_id": q.id, "qtype": q.qtype, "n_items": len(items),
                   "hit": hit, "recall_at_k": metrics.recall_at_k(ids, q.gold_ids, "session", a.k),
                   "hits": ids, "gold": sorted(q.gold_ids), "ingest_s": round(ingest_s, 1), "recall_ms": round(ms)}
            rows.append(row)
            print(f"[{cname:15}] {q.id[:14]} {q.qtype[:22]:22} hit={hit!s:5} recall@k={row['recall_at_k']:.2f} "
                  f"items={len(items)} ingest={ingest_s:.0f}s recall={ms:.0f}ms", flush=True)
            with open(os.path.join(a.out, "probe.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        totals[cname] = hits_n / max(1, len(qs))
        print(f"== {cname}: hit@{a.k} = {hits_n}/{len(qs)} = {totals[cname]:.3f}", flush=True)
    os.environ.clear()
    os.environ.update(base_env)
    print("totals:", json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
