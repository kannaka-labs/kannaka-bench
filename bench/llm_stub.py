"""A stub OpenAI-compatible chat endpoint for retrieval-only competitor runs.

Supermemory's self-hosted server calls an LLM for every ingested document
(a `maintain-container-description` workflow step, 30 s timeout) even in
`superrag` task mode, where the LLM output does not touch the chunk index
that `searchMode=documents` searches. On a saturated CPU box that step made
ingest ~8 s/document and never finished. This stub answers instantly with a
fixed, harmless completion so the retrieval-only row measures retrieval.

It is NOT a substitute for their memory engine: `supermemory_mem` (fact
extraction, `memories` search) needs a real model and is run separately,
with its cost in the table. Every request's purpose (first line of the last
message) is appended to BENCH_LLM_STUB_LOG so what was stubbed is on record.

  python -m bench.llm_stub --port 11438
  OPENAI_BASE_URL=http://127.0.0.1:11438/v1 OPENAI_API_KEY=stub OPENAI_MODEL=stub supermemory-server
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = os.environ.get("BENCH_LLM_STUB_LOG", "")
TEXT = os.environ.get("BENCH_LLM_STUB_TEXT", "Conversation turns between a user and an assistant.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [{"id": "stub", "object": "model", "owned_by": "kannaka-bench"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad json"})
        msgs = req.get("messages") or []
        last = (msgs[-1].get("content") if msgs and isinstance(msgs[-1], dict) else "") or ""
        if isinstance(last, list):
            last = " ".join(str(p.get("text", "")) for p in last if isinstance(p, dict))
        wants_json = (req.get("response_format") or {}).get("type", "").startswith("json")
        content = "{}" if wants_json else TEXT
        if LOG:
            try:
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": round(time.time(), 1), "model": req.get("model"), "json": wants_json,
                                        "purpose": str(last)[:120].replace("\n", " ")}) + "\n")
            except OSError:
                pass
        self._json(200, {"id": "stub", "object": "chat.completion", "created": int(time.time()), "model": req.get("model", "stub"),
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11438)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
