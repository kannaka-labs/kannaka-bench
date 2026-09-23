"""Does the medium store the embedding it was given? Export a store's wavefronts
and compare each stored vector to a fresh embedding of the same content from
the embed server: cosine 1.0 means the store is faithful; anything lower
means the medium transforms vectors before ranking, and cosine-based
baselines are not comparable to it turn for turn.

  python -m bench.vector_fidelity --store /tmp/kannaka-probe-*/xioff/<qid>/kannaka \
      --bin <kannaka> --embed-url http://127.0.0.1:11436 [--sample 40]

CAVEAT (2026-09-17): `kannaka export-json` dumps the legacy FLAT medium
(10 000-dim, the consciousness-metrics representation), not the chiral
hemisphere rows that recall actually ranks over - so this tool's "stored"
vectors are the wrong representation for that question (mean cosine ~0 is
expected). Kept for the flat path and until an export of chiral rows exists;
`add_wavefront` stores the raw zero-padded embedding, verified by reading.
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
    with urllib.request.urlopen(req, timeout=300) as r:
        return np.array(json.load(r)["embeddings"], dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--bin", default=os.environ.get("KANNAKA_BIN", "kannaka"))
    ap.add_argument("--embed-url", default=os.environ.get("BENCH_OLLAMA_URL", "http://127.0.0.1:11436"))
    ap.add_argument("--sample", type=int, default=40)
    a = ap.parse_args(argv)
    stores = sorted(glob.glob(a.store), key=os.path.getmtime)
    if not stores:
        sys.exit("no store")
    store = stores[-1]
    env = dict(os.environ, KANNAKA_DATA_DIR=store, KANNAKA_NATS_URL="nats://127.0.0.1:1", KANNAKA_ENCODER="ollama",
               KANNAKA_ENCODER_URL=a.embed_url, KANNAKA_ENCODER_MODEL="all-minilm", KANNAKA_ENCODER_DIM="384")
    r = subprocess.run([a.bin, "export-json"], env=env, capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace")
    line = next((l for l in r.stdout.split("\n") if l.strip().startswith("[")), "[]")
    mems = json.loads(line)
    print(f"store {store}: {len(mems)} memories exported")
    rows = [m for m in mems if m.get("vector") and (m.get("content") or "").strip()]
    rows = rows[: a.sample]
    stored = np.array([m["vector"] for m in rows], dtype=np.float32)
    fresh = embed(a.embed_url, [m["content"] for m in rows])
    dims = stored.shape[1], fresh.shape[1]
    n = min(dims)
    s = stored[:, :n]; f = fresh[:, :n]
    s = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-9)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    cos = (s * f).sum(axis=1)
    print(f"stored dims {dims[0]}, fresh dims {dims[1]}; stored-vs-fresh cosine over {len(rows)} memories: "
          f"mean {cos.mean():.4f} min {cos.min():.4f} max {cos.max():.4f}")
    extra = float(np.abs(stored[:, n:]).sum()) if stored.shape[1] > n else 0.0
    print(f"mass beyond dim {n} in stored vectors: {extra:.4f}")
    worst = np.argsort(cos)[:3]
    for i in worst:
        print(f"  cos={cos[i]:.4f} {rows[i]['content'][:90]!r}")
    # ranking check: exact cosine (fresh) vs cosine over stored vectors for one query
    q = "What degree did I graduate with?"
    qv = embed(a.embed_url, [q])[0][:n]; qv /= max(np.linalg.norm(qv), 1e-9)
    of = np.argsort(-(f @ qv))[:5]; os_ = np.argsort(-(s @ qv))[:5]
    print(f"top-5 by fresh cosine: {list(of)}; by stored cosine: {list(os_)} (indices into the sample)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
