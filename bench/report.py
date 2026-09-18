"""Aggregate a run into one table — losses in the same table as wins.

  python -m bench.report results/<run-id>/
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from . import metrics


def load_rows(run_dir: str) -> list[dict]:
    rows = []
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["adapter"]].append(r)
    out = {}
    for name, rs in by.items():
        ok = [r for r in rs if "error" not in r]
        errors = len(rs) - len(ok)
        if not ok:
            out[name] = {"n": 0, "errors": errors}
            continue
        k = ok[0]["k"]
        # Rows with no gold evidence (ConvoMem abstention: the right answer is
        # "no information") have no retrieval score; they count in n but not
        # in hit/recall/MRR, and their type shows n/a.
        scored = [r for r in ok if r.get("gold")] or ok
        out[name] = {
            "n": len(ok), "errors": errors, "k": k, "unscored": len(ok) - len([r for r in ok if r.get("gold")]),
            "any_hit_at_k": sum(1 for r in scored if r["any_hit_at_k"]) / len(scored),
            "recall_at_k": sum(r["recall_at_k"] for r in scored) / len(scored),
            "mrr": sum(r["mrr"] for r in scored) / len(scored),
            "evidence_coverage_at_k": (lambda ev: (sum(ev) / len(ev)) if ev else float("nan"))(
                [r["evidence_coverage_at_k"] for r in scored if r.get("evidence_coverage_at_k") is not None]),
            "recall_p50_ms": metrics.percentile([r["recall_ms"] for r in ok], 0.5),
            "recall_p95_ms": metrics.percentile([r["recall_ms"] for r in ok], 0.95),
            "ingest_ms_per_item": sum(r["ingest_ms_per_item"] for r in ok) / len(ok),
            "footprint_bytes_per_item": sum(r["footprint_bytes"] / max(1, r["n_items"]) for r in ok) / len(ok),
            "by_qtype": {},
        }
        types = defaultdict(list)
        for r in ok:
            types[r.get("qtype") or "?"].append(r)
        for t, trs in sorted(types.items()):
            sc = [r for r in trs if r.get("gold")]
            out[name]["by_qtype"][t] = {"n": len(trs), "any_hit_at_k": (sum(1 for r in sc if r["any_hit_at_k"]) / len(sc)) if sc else float("nan")}
    return out


def render(run_dir: str) -> str:
    rows = load_rows(run_dir)
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    s = summarize(rows)
    ds = manifest.get("dataset", {})
    lines = [
        f"## {manifest.get('run_id')} — {ds.get('dataset')} (sha256 {str(ds.get('sha256', ''))[:12]}…), k={manifest.get('k')}, "
        f"{manifest.get('stores')} stores, host {manifest.get('host')}, commit {str(manifest.get('commit', ''))[:8]}",
        "",
        "| adapter | n | hit@k | recall@k | evid@k | MRR | recall p50 ms | p95 ms | ingest ms/item | bytes/item | errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    best = max((v.get("any_hit_at_k", -1) for v in s.values()), default=-1)
    for name, v in sorted(s.items(), key=lambda kv: -kv[1].get("any_hit_at_k", -1)):
        if not v.get("n"):
            lines.append(f"| {name} | 0 | — | — | — | — | — | — | — | {v.get('errors', 0)} |")
            continue
        mark = " ◀" if v["any_hit_at_k"] == best and len(s) > 1 else ""
        lines.append(f"| {name}{mark} | {v['n']} | {v['any_hit_at_k']:.3f} | {v['recall_at_k']:.3f} | {v.get('evidence_coverage_at_k', float('nan')):.3f} | {v['mrr']:.3f} | "
                     f"{v['recall_p50_ms']:.0f} | {v['recall_p95_ms']:.0f} | {v['ingest_ms_per_item']:.1f} | "
                     f"{v['footprint_bytes_per_item']:.0f} | {v['errors']} |")
    types = sorted({t for v in s.values() for t in v.get("by_qtype", {})})
    if types and len(s) > 1:
        lines += ["", "hit@k by question type:", "", "| type | " + " | ".join(s.keys()) + " |", "|---|" + "---|" * len(s)]
        for t in types:
            cells = [f"{v.get('by_qtype', {}).get(t, {}).get('any_hit_at_k', float('nan')):.3f} (n={v.get('by_qtype', {}).get(t, {}).get('n', 0)})" for v in s.values()]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(sys.argv[1]))
