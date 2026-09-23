"""kannaka — the binary, one HRM store per run.

Bulk path (kannaka-memory #973): `kannaka remember --batch items.ndjson`
loads every item in one process and prints one id per line in order;
`kannaka recall --batch queries.ndjson` answers many queries from one
loaded store. Without those flags (an older binary) the adapter falls back
to one spawn per item — ~620 ms each on Linux, the reason the flags exist.
Ids kannaka prints are kept so hits map back to our item ids. Swarm
publishing is disabled by pointing the binary at an unreachable NATS URL —
a benchmark must never write into the production swarm (it did, on the
first probe, 2026-09-17). Shipped defaults otherwise.
"""
from __future__ import annotations

import json
import re
import os
import time
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit, dir_bytes

BIN = os.environ.get("KANNAKA_BIN", "kannaka")


class KannakaAdapter(Adapter):
    name = "kannaka"

    def __init__(self, bin_path: str = BIN, timeout_s: float = 60.0):
        self.bin = bin_path
        self.timeout_s = timeout_s
        self.dir = None
        self.env = None
        self.by_kid: dict[str, str] = {}      # kannaka memory id -> item id
        self.text_of: dict[str, str] = {}
        self.version = None
        self.batch = os.environ.get("KANNAKA_BATCH", "auto")   # auto | 1 | 0
        self.batch_ok = None                                    # learned on first use
        # E-L3c: a JSON map item_id -> {"expires": iso, ...} produced by
        # experiments/laya_reflex/e_l3c_supersede.py. At ingest every mapped item
        # gets `expires`; with BENCH_DROP_EXPIRED=1 recall drops hits whose
        # expires <= the question's asked_at (the rule kannaka-memory does not
        # apply itself yet; see the E-L3c entry in RESULTS.md).
        self.supersede: dict = {}
        m = os.environ.get("BENCH_SUPERSEDE_MAP")
        if m:
            with open(m, encoding="utf-8") as f:
                self.supersede = json.load(f).get("map", {})
        self.drop_expired = os.environ.get("BENCH_DROP_EXPIRED", "0") == "1"
        self.expires_of: dict = {}   # item id -> aware datetime, for the recall-time filter

    #: `[beam] scored 512 of 1671 memories (30.6%) -> 5 results`
    _BEAM_RE = re.compile(r"\[beam\] scored (\d+) of (\d+) memories")

    def _note_stderr(self, err: str) -> None:
        """Record the attention-beam trace instead of discarding it.

        Both runners captured stderr and used it only on failure, so a
        successful sparse recall left no evidence it had happened. A bench arm
        that cannot show whether its feature fired cannot be compared against
        one that did — the first beam run was uninterpretable for exactly this
        reason. Parsed here, surfaced in the manifest by run.py.
        """
        if not err:
            return
        for m in self._BEAM_RE.finditer(err):
            scored, total = int(m.group(1)), int(m.group(2))
            self.beam["recalls"] += 1
            self.beam["scored"] += scored
            self.beam["total"] += total

    def _batch_run(self, args: list[str], timeout: float):
        """One batch subprocess; returns (returncode, stdout, stderr). Stubbed in tests."""
        r = subprocess.run([self.bin] + args, env=self.env, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        err = r.stderr or ""
        self._note_stderr(err)
        return r.returncode, r.stdout or "", err

    def _run(self, args: list[str]) -> str:
        r = subprocess.run([self.bin] + args, env=self.env, capture_output=True, text=True,
                           timeout=self.timeout_s, encoding="utf-8", errors="replace")
        self._note_stderr(r.stderr or "")
        if r.returncode != 0:
            raise RuntimeError(f"kannaka {args[0]} failed: {r.stderr.strip()[:200]}")
        return r.stdout

    def beam_stats(self) -> dict:
        """What the beam actually did, or that it never fired.

        `recalls == 0` with KANNAKA_RECALL_BEAM set is a real finding, not a
        blank: it means the store had no skip links to walk (they are written
        only by dream consolidation) and every recall silently fell back to a
        dense scan.
        """
        b = dict(self.beam)
        b["mean_coverage"] = round(b["scored"] / b["total"], 4) if b["total"] else None
        b["enabled_env"] = os.environ.get("KANNAKA_RECALL_BEAM", "")
        return b

    def open(self, run_dir: str) -> None:
        self.dir = os.path.join(run_dir, "kannaka")
        os.makedirs(self.dir, exist_ok=True)
        self.beam = {"recalls": 0, "scored": 0, "total": 0}
        #: None = untested, True/False = this binary does/doesn't take `--at`.
        self.supports_at = None
        self.env = dict(os.environ, KANNAKA_DATA_DIR=self.dir,
                        KANNAKA_NATS_URL="nats://127.0.0.1:1",   # off the swarm, always
                        # Facets default OFF since 2026-09-22: on longmemeval_s the facets-off chiral
                        # arm reproduced retrieval exactly at 2.8x lower recall latency and
                        # 4x less disk (RESULTS.md). OFF is also the kannaka-memory binary's
                        # own default; set KANNAKA_FACET_DECOMPOSE=1 to run the facets arm.
                        KANNAKA_FACET_DECOMPOSE=os.environ.get("KANNAKA_FACET_DECOMPOSE", "0"))
        try:
            self.version = subprocess.run([self.bin, "--version"], capture_output=True, text=True,
                                          timeout=10).stdout.strip()
        except Exception:
            self.version = "unknown"
        self.by_kid.clear()
        self.text_of.clear()

    def _ingest_batch(self, items: list[MemoryItem]) -> bool:
        """One process for the whole list. Returns False if the binary has no
        --batch (fall back), raises on any other failure."""
        path = os.path.join(self.dir, "ingest.ndjson")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                row = {"content": it.text[:4000]}
                if it.when is not None:
                    iso = it.when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    row["observed"] = iso
                    row["effective"] = iso
                sup = self.supersede.get(it.id)
                if sup:
                    row["expires"] = sup["expires"]
                    self.expires_of[it.id] = datetime.strptime(sup["expires"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rc, out, err = self._batch_run(["remember", "--batch", path], max(self.timeout_s, 2.0 * len(items) + 60))
        if "unknown flag" in err and "--batch" in err:
            return False
        ids = [l.strip() for l in out.split("\n") if l.strip()]
        if rc not in (0, 1) or len(ids) != len(items):
            raise RuntimeError(f"kannaka remember --batch: rc={rc}, {len(ids)} lines for {len(items)} items: "
                               f"{err.strip()[:200]}")
        for it, kid in zip(items, ids):
            if len(kid) == 36 and kid.count("-") == 4:
                self.by_kid[kid] = it.id
            self.text_of[it.id] = it.text
        return True

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        items = list(items)
        if not items:
            return
        if self.batch != "0" and self.batch_ok is not False:
            if self._ingest_batch(items):
                self.batch_ok = True
                return
            self.batch_ok = False   # older binary: per-item from here on
        for it in items:
            args = ["remember", it.text[:4000]]
            if it.when is not None:
                iso = it.when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                args += ["--observed", iso, "--effective", iso]
            out = self._run(args)
            kid = None
            for line in reversed(out.strip().split("\n")):
                line = line.strip()
                if len(line) == 36 and line.count("-") == 4:
                    kid = line
                    break
            if kid:
                self.by_kid[kid] = it.id
            self.text_of[it.id] = it.text

    def _expired(self, iid: str, when) -> bool:
        """E-L3c: a memory superseded before the question was asked is not an answer."""
        if not self.drop_expired or when is None:
            return False
        exp = self.expires_of.get(iid)
        if exp is None:
            return False
        try:
            w = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        except AttributeError:
            return False
        return exp <= w

    def _parse_rows(self, rows, k: int, when=None) -> list[RecallHit]:
        hits = []
        for r in rows:
            kid = r.get("id")
            iid = self.by_kid.get(kid)
            if iid is None:
                iid = f"kannaka:{kid}"   # a memory kannaka made itself (a dream, a merge)
            if self._expired(iid, when):
                continue
            hits.append(RecallHit(id=iid, score=float(r.get("similarity") or 0.0), text=(r.get("content") or "")[:200]))
        return hits[:k]

    def recall_many(self, queries: list[str], k: int, whens=None) -> list[list[RecallHit]]:
        """Many queries, one process (recall --batch); falls back to one spawn each.

        `at` goes per ROW, not per process: each question has its own date, and
        one flag for the whole file could not express that.
        """
        whens = list(whens) if whens is not None else [None] * len(queries)
        if self.batch != "0" and self.batch_ok is not False:
            path = os.path.join(self.dir, "queries.ndjson")
            with open(path, "w", encoding="utf-8") as f:
                for q, w in zip(queries, whens):
                    # over-fetch when expired rows may be dropped, so k survive (k+5, see #1045)
                    row = {"query": q[:1000], "top_k": k + 5 if self.drop_expired else k}
                    at = self._at_args(w)
                    if at:
                        row["at"] = at[1]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rc, stdout, err = self._batch_run(["recall", "--batch", path], max(self.timeout_s, 1.0 * len(queries) + 60))
            if not ("unknown flag" in err and "--batch" in err) and rc == 0:
                out = []
                for line in stdout.split("\n"):
                    line = line.strip()
                    if line.startswith("["):
                        try:
                            out.append(self._parse_rows(json.loads(line), k, whens[len(out)] if len(out) < len(whens) else None))
                        except ValueError:
                            out.append([])
                if len(out) == len(queries):
                    self.batch_ok = True
                    return out
            self.batch_ok = False
        return [self.recall(q, k, w) for q, w in zip(queries, whens)]

    @staticmethod
    def _at_args(when) -> list[str]:
        """`--at` for the question's own date, or nothing.

        Temporal weight decays from a memory's `observed_at` to "now", and the
        result is clamped up to the superseded floor — a clamp that binds at
        two half-lives (360 days by default). Every corpus here is dated 2023,
        so against the wall clock EVERY candidate returns the floor, the
        temporal factor becomes a constant multiplier, and it ranks nothing.
        Scoring as of the moment the question was asked is what makes the
        factor mean anything, and is the question a real agent asks anyway.

        Older binaries reject the flag; `_supports_at` falls back once.
        """
        if when is None:
            return []
        try:
            iso = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (AttributeError, ValueError):
            return []
        return ["--at", iso]

    def recall(self, query: str, k: int, when=None) -> list[RecallHit]:
        # E-L3c: over-fetch when expired rows may be dropped, so k survive.
        # (This single-question path is what run.py uses for LongMemEval — one
        # question per store — so the drop has to live here, not only in
        # recall_many; the first E-L3c retrieval arms were identical because
        # it did not.)
        # Over-fetch modestly (k+5) so k survive the drop. The retry below is a
        # guard only: the "no rows at --top-k 30" that kannaka-memory #1045 blamed on
        # the binary was THIS parser splitting the JSON line at a U+2028 inside a
        # memory's content (Python's splitlines() honours it; serde_json emits it
        # raw). Rows are split on "\n" only now; see the regression test.
        fetch = k + 5 if self.drop_expired else k
        args = ["recall", query[:1000], "--top-k", str(fetch)]
        if self.supports_at is not False:
            args += self._at_args(when)
        try:
            out = self._run(args)
        except RuntimeError as e:
            # A binary predating --at exits 2 with "unknown flag". Drop the
            # flag, record it, and keep going rather than failing the run —
            # but say so, because the arm is then NOT measuring what it claims.
            if self.supports_at is None and "--at" in str(e):
                self.supports_at = False
                print(f"[kannaka] binary rejects --at; temporal scoring will use the WALL CLOCK "
                      f"and the temporal exponent cannot rank on this corpus: {str(e)[:120]}",
                      file=sys.stderr)
                out = self._run(["recall", query[:1000], "--top-k", str(fetch)])
            else:
                raise
        else:
            if self.supports_at is None and when is not None:
                self.supports_at = True
        hits = []
        # split("\n"), never splitlines(): a JSON line is one line even when a
        # memory's text carries U+2028/U+2029/NEL (LongMemEval's ShareGPT turns do).
        for line in reversed(out.strip().split("\n")):
            line = line.strip()
            if line.startswith("["):
                try:
                    rows = json.loads(line)
                except ValueError:
                    rows = []
                for r in rows:
                    kid = r.get("id")
                    iid = self.by_kid.get(kid)
                    if iid is None:
                        # a memory kannaka made itself (a dream, a merge): keep it,
                        # it just cannot score
                        iid = f"kannaka:{kid}"
                    if self._expired(iid, when):
                        continue
                    hits.append(RecallHit(id=iid, score=float(r.get("similarity") or 0.0),
                                          text=(r.get("content") or "")[:200]))
                break
        if not hits and fetch != k:
            print(f"[kannaka] recall returned no rows at --top-k {fetch}; retrying at {k}",
                  file=sys.stderr)
            saved, self.drop_expired = self.drop_expired, False
            try:
                hits = self.recall(query, k, when)
            finally:
                self.drop_expired = saved
            return [h for h in hits if not self._expired(h.id, when)][:k]
        return hits[:k]

    def _memory_count(self):
        out = self._run(["status"])
        m = re.search(r'"total_memories":\s*(\d+)', out or "")
        return int(m.group(1)) if m else None

    def consolidate(self) -> dict:
        """One dream cycle over the freshly ingested store (`kannaka dream
        --mode deep`, BENCH_DREAM_MODE=lite for the quick pass,
        BENCH_DREAM_CHIRAL=<eta> for spiral dynamics). The paper's central
        claim — a memory that joins evidence before the question — lives here;
        the row records the cost (ms, memories minted) beside whatever it did
        to recall."""
        mode = os.environ.get("BENCH_DREAM_MODE", "deep")
        args = ["dream", "--mode", mode]
        chiral = os.environ.get("BENCH_DREAM_CHIRAL")
        if chiral:
            args += ["--chiral", chiral]
        before = self._memory_count()
        t0 = time.perf_counter()
        rc, out, err = self._batch_run(args, max(self.timeout_s, 1800.0))
        ms = (time.perf_counter() - t0) * 1000.0
        after = self._memory_count()
        return {"mode": mode, "chiral": chiral, "ms": round(ms), "rc": rc, "memories_before": before,
                "memories_after": after, "minted": (after - before) if (before is not None and after is not None) else None,
                "note": (err or "")[-200:] if rc != 0 else ""}

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir) if self.dir else 0

    def close(self) -> None:
        pass


class KannakaMinilmAdapter(KannakaAdapter):
    """kannaka with the all-MiniLM-L6-v2 encoder (via ollama) instead of the
    shipped default. A fresh store's `.encoder` reads `hash:384:42`: a hashing
    encoder, no semantics — which is what the first longmemeval_s row (hit@k
    0.533 vs 0.967 for MiniLM cosine) actually measured. This row asks the
    fair question: same encoder family, does the medium rank better or worse
    than exact cosine? Needs `ollama pull all-minilm` on the box.
    BENCH_OLLAMA_URL overrides the endpoint."""
    name = "kannaka_minilm"

    def open(self, run_dir: str) -> None:
        super().open(run_dir)
        self.env.update({
            "KANNAKA_ENCODER": "ollama",
            "KANNAKA_ENCODER_URL": os.environ.get("BENCH_OLLAMA_URL", "http://127.0.0.1:11434"),
            "KANNAKA_ENCODER_MODEL": os.environ.get("BENCH_OLLAMA_EMBED_MODEL", "all-minilm"),
            "KANNAKA_ENCODER_DIM": "384",
        })


class KannakaBgeAdapter(KannakaMinilmAdapter):
    """kannaka with BAAI/bge-base-en-v1.5 (768-d) through the bench embed server
    (`python -m bench.embed_server --port 11439` with
    BENCH_EMBED_MODEL=BAAI/bge-base-en-v1.5) — Supermemory's default encoder,
    paired with `vector_bge` on the same weights. BENCH_BGE_URL overrides."""
    name = "kannaka_bge"

    def open(self, run_dir: str) -> None:
        super().open(run_dir)
        self.env.update({
            "KANNAKA_ENCODER": "ollama",
            "KANNAKA_ENCODER_URL": os.environ.get("BENCH_BGE_URL", "http://127.0.0.1:11439"),
            "KANNAKA_ENCODER_MODEL": "bge-base-en-v1.5",
            "KANNAKA_ENCODER_DIM": "768",
        })
