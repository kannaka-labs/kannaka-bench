"""Merge the shard directories of one run (`run.py --shard I/N`) into one run dir.

  python -m bench.merge results/<run> results/<run>-shard0 results/<run>-shard1 ...

Rows are concatenated; a duplicated (adapter, question) pair or shards that
disagree on dataset / adapters / k / commit / shard count is an error, not a
silent merge. The merged manifest keeps every shard's own manifest under
`shards`, so per-shard hosts, wall times and binaries stay on the record.
"""
from __future__ import annotations

import json
import os
import sys

MUST_AGREE = ("adapters", "k", "session_cap", "limit", "commit", "stores_total")


def merge(out_dir: str, shard_dirs: list[str]) -> dict:
    if not shard_dirs:
        raise ValueError("no shard dirs")
    manifests, rows, seen = [], [], set()
    for d in shard_dirs:
        m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        manifests.append(m)
        for line in open(os.path.join(d, "results.jsonl"), encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["adapter"], r["question_id"])
            if key in seen:
                raise ValueError(f"duplicate row {key} (in {d})")
            seen.add(key)
            rows.append(r)
    first = manifests[0]
    for m in manifests[1:]:
        for f in MUST_AGREE:
            if m.get(f) != first.get(f):
                raise ValueError(f"shards disagree on {f}: {first.get(f)!r} vs {m.get(f)!r}")
        if (m.get("dataset") or {}).get("sha256") != (first.get("dataset") or {}).get("sha256"):
            raise ValueError("shards ran different dataset files")
    shards = sorted({m.get("shard") for m in manifests}, key=str)
    if len(shards) != len(manifests):
        raise ValueError(f"shard specs repeat: {[m.get('shard') for m in manifests]}")
    merged = dict(first)
    merged.update({
        "run_id": os.path.basename(os.path.normpath(out_dir)),
        "stores": sum(m.get("stores", 0) for m in manifests),
        "shard": None,
        "started_at": min(m.get("started_at", "") for m in manifests),
        "finished_at": max(m.get("finished_at", "") for m in manifests),
        # shards ran side by side: the run took as long as its slowest shard
        "wall_s": max(m.get("wall_s", 0) for m in manifests),
        "merged_from": [os.path.basename(os.path.normpath(d)) for d in shard_dirs],
        "shards": manifests,
    })
    for f in ("adapter_versions", "adapter_bins"):
        vals = {}
        for m in manifests:
            vals.update(m.get(f) or {})
        merged[f] = vals
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=1)
    return {"rows": len(rows), "stores": merged["stores"], "stores_total": merged.get("stores_total")}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    print(merge(argv[0], argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
