"""Explain one miss: replay a question against its leftover kannaka store with
KANNAKA_RECALL_TRACE=1 and compare what the medium ranked to exact cosine
over the same embeddings (via the embed server), so a ranking loss can be
attributed to a stage rather than guessed at.

  python -m bench.trace_miss --run ~/kannaka-bench-results/<run> --adapter kannaka_minilm \
      --question <question_id> --embed-url http://127.0.0.1:11436 --bin <kannaka>

Needs the store dir the run left under /tmp/kannaka-bench-*/<question_id>/kannaka.
Prints: the trace lines (left/right sim + rs per candidate), the final top-k
with hemisphere of origin, and exact-cosine rank of the gold session's best
turn among all items.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np


def embed(url: str, texts: list[str]) -> np.ndarray:
    req = urllib.request.Request(url.rstrip("/") + "/api/embed", data=json.dumps({"model": "all-minilm", "input": texts}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return np.array(json.load(r)["embeddings"], dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapter", default="kannaka_minilm")
    ap.add_argument("--question", default=None, help="default: the first miss of that adapter")
    ap.add_argument("--dataset", default="longmemeval_s")
    ap.add_argument("--embed-url", default=os.environ.get("BENCH_OLLAMA_URL", "http://127.0.0.1:11436"))
    ap.add_argument("--bin", default=os.environ.get("KANNAKA_BIN", "kannaka"))
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(os.path.join(a.run, "results.jsonl"), encoding="utf-8") if l.strip()]
    mine = [r for r in rows if r["adapter"] == a.adapter and "error" not in r]
    row = next((r for r in mine if r["question_id"] == a.question), None) if a.question else next((r for r in mine if not r["any_hit_at_k"]), None)
    if not row:
        sys.exit("no such question / no miss to explain")
    qid = row["question_id"]
    stores = glob.glob(f"/tmp/kannaka-bench-*/{qid}/kannaka/kannaka.hrm")
    if not stores:
        sys.exit(f"no leftover store for {qid} under /tmp/kannaka-bench-*")
    store = os.path.dirname(sorted(stores, key=os.path.getmtime)[-1])
    print(f"question {qid} ({row['qtype']}); gold {row['gold']}; kannaka hits {[h.split('#')[0] for h in row['hits']]}")
    print(f"store: {store}")

    from .datasets import longmemeval
    qs, _ = longmemeval.load(a.dataset, limit=None)
    q = next(x for x in qs if x.id == qid)
    print(f"question text: {q.question!r}")

    # exact cosine over the same embeddings
    texts = [it.text[:4000] for it in q.items]
    M = embed(a.embed_url, texts)
    qv = embed(a.embed_url, [q.question])[0]
    scores = M @ qv
    order = np.argsort(-scores)
    ranked_ids = [q.items[i].id for i in order]
    gold_ranks = [i + 1 for i, iid in enumerate(ranked_ids) if iid.split("#")[0] in q.gold_ids]
    print(f"exact cosine: best gold turn at rank {gold_ranks[0] if gold_ranks else 'none'}; "
          f"top-{a.k} sessions {[x.split('#')[0] for x in ranked_ids[:a.k]]}; top scores {[round(float(scores[i]),3) for i in order[:a.k]]}")

    # the medium, traced
    env = dict(os.environ, KANNAKA_DATA_DIR=store, KANNAKA_NATS_URL="nats://127.0.0.1:1", KANNAKA_RECALL_TRACE="1",
               KANNAKA_ENCODER="ollama", KANNAKA_ENCODER_URL=a.embed_url, KANNAKA_ENCODER_MODEL="all-minilm",
               KANNAKA_ENCODER_DIM="384")
    r = subprocess.run([a.bin, "recall", q.question[:1000], "--top-k", str(a.k)], env=env, capture_output=True, text=True,
                       timeout=300, encoding="utf-8", errors="replace")
    trace = [l for l in r.stderr.splitlines() if "recall-trace" in l]
    print(f"--- trace ({len(trace)} lines):")
    for l in trace[:60]:
        print("  " + l[:170])
    final = [l for l in r.stdout.split("\n") if l.strip().startswith("[")]
    if final:
        hits = json.loads(final[-1])
        print("--- final top-k (content, similarity):")
        for h in hits:
            print(f"  sim={h.get('similarity'):.4f} {(h.get('content') or '')[:100]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
