"""Post-hoc turn-level evidence coverage for finished runs (runs made before the
runner recorded `evidence_coverage_at_k`): of a question's answer-bearing turns
(item.meta.has_answer) inside its gold sessions, the fraction found in the top-k.

  python -m bench.evidence_coverage results/<run>/ [results/<run2>/ ...]

Prints per adapter: mean coverage, fraction of questions with ALL evidence turns
in the top-k, and both by question type. Loads the run's dataset from its
manifest (LongMemEval variants and ConvoMem).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

_cache: dict = {}


def questions_for(manifest: dict):
    ds = manifest["dataset"]
    name = ds.get("dataset") or ""
    limit = manifest.get("limit")
    key = (name, ds.get("context_size"), limit)
    if key in _cache:
        return _cache[key]
    if name.startswith("convomem"):
        from .datasets import convomem
        qs, _ = convomem.load(context=int(ds.get("context_size") or 50), limit=limit)
    else:
        from .datasets import longmemeval
        qs, _ = longmemeval.load(name, limit=limit)
    _cache[key] = {q.id: q for q in qs}
    return _cache[key]


def main(argv=None):
    for run in (argv or sys.argv[1:]):
        manifest = json.load(open(os.path.join(run, "manifest.json"), encoding="utf-8"))
        qmap = questions_for(manifest)
        rows = [json.loads(l) for l in open(os.path.join(run, "results.jsonl"), encoding="utf-8") if l.strip()]
        by_ad = defaultdict(list)
        for r in rows:
            if "error" in r or not r.get("gold"):
                continue
            q = qmap.get(r["question_id"])
            if not q:
                continue
            evidence = {it.id for it in q.items if (it.meta or {}).get("has_answer")
                        and (it.id.rpartition("#")[0] if "#" in it.id else it.id) in q.gold_ids}
            if not evidence:
                continue
            k = r["k"]
            found = len(set(r["hits"][:k]) & evidence)
            by_ad[r["adapter"]].append((r["qtype"], found / len(evidence), found == len(evidence), len(evidence)))
        print(f"== {os.path.basename(os.path.normpath(run))}")
        for ad, xs in by_ad.items():
            cov = sum(x[1] for x in xs) / len(xs)
            allc = sum(x[2] for x in xs) / len(xs)
            by = defaultdict(list)
            for x in xs:
                by[x[0]].append(x)
            print(f"   {ad:14} n={len(xs)} evidence-coverage@k={cov:.3f} all-evidence@k={allc:.3f} evidence turns/q={sum(x[3] for x in xs) / len(xs):.1f}")
            print("      by type (coverage / all):", {t: "%.2f / %.2f" % (sum(v[1] for v in vs) / len(vs), sum(v[2] for v in vs) / len(vs)) for t, vs in sorted(by.items())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
