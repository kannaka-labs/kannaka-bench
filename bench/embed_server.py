"""A tiny embedding server speaking ollama's `/api/embed` shape, backed by
sentence-transformers in-process — the SAME weights the vector baseline uses.

Why: kannaka's only non-hash encoder talks to ollama, and ollama's MiniLM
embed cost 1.3–2.3 s per call on both lab boxes (2026-09-17), while the same
model in-process costs ~15 ms. Pointing kannaka at this server gives the
`kannaka_minilm` row identical embeddings to `vector_numpy`, so whatever the
table then says is about the medium and nothing else.

  python -m bench.embed_server --port 11436
  KANNAKA_ENCODER=ollama KANNAKA_ENCODER_URL=http://127.0.0.1:11436 ...

POST /api/embed  {"model": "...", "input": "text" | ["t1", "t2"]}
  -> {"model": "...", "embeddings": [[...], ...]}
GET  /api/version -> {"version": "kannaka-bench-embed"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("BENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None
_lock = threading.Lock()


def model():
    global _model
    if _model is None:
        # Single-text calls are latency-bound: torch grabbing every core on a
        # busy box cost 428 ms per call; two threads cost 74 ms (debain2, load 6).
        try:
            import torch
            torch.set_num_threads(int(os.environ.get("BENCH_EMBED_THREADS", "2")))
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL)
    return _model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/version"):
            return self._json(200, {"version": "kannaka-bench-embed"})
        if self.path.startswith("/api/tags"):
            return self._json(200, {"models": [{"name": MODEL}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/api/embed"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad json"})
        inp = req.get("input", req.get("prompt", ""))
        texts = inp if isinstance(inp, list) else [inp]
        texts = [str(t) for t in texts]
        if not texts or not any(t.strip() for t in texts):
            return self._json(400, {"error": "empty input"})
        with _lock:
            vecs = model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._json(200, {"model": req.get("model", MODEL), "embeddings": [v.tolist() for v in vecs]})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11436)
    a = ap.parse_args(argv)
    model()  # load once, up front
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"embed server: {MODEL} on http://{a.host}:{a.port}/api/embed", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
