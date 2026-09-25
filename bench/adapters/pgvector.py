"""pgvector — the same MiniLM embeddings in Postgres, the way most teams ship
"agent memory" today.

  BENCH_PG_DSN            postgresql://bench:bench@127.0.0.1:5433/bench
  BENCH_PG_EF_SEARCH      hnsw.ef_search for the `pgvector` arm (default: the
                          server's own default, 40 — not tuned on the test set)

Two adapter names:

  `pgvector`        HNSW index (`vector_cosine_ops`, pgvector's defaults m=16,
                    ef_construction=64), created BEFORE the inserts so every
                    turn lands in a live index — the incremental shape an agent
                    writing memories one at a time has, not a bulk load.
  `pgvector_exact`  no index: Postgres scans every row and returns the exact
                    cosine top-k. It must reproduce `vector_numpy` to the row;
                    it is the control that says the plumbing (encoding, vector
                    literals, the distance operator, id attribution) is right,
                    so any gap in the HNSW arm is the index and nothing else.

Encoder held fixed: the embeddings come from the SAME sentence-transformers
all-MiniLM-L6-v2 weights `vector_numpy` loads in-process, normalised, so the
row measures the database and its index, not the embedding. Encoding time is
inside ingest and recall, as it is for `vector_numpy`.

One table per store (LongMemEval: one per question), dropped on close after
its footprint (`pg_total_relation_size`: heap + TOAST + index) is read, so a
30-question run leaves nothing behind in the database.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable

from .base import Adapter, MemoryItem, RecallHit

DSN = os.environ.get("BENCH_PG_DSN", "postgresql://bench:bench@127.0.0.1:5433/bench")


def vec_literal(v) -> str:
    """pgvector's text input form. repr-precision floats: a rounded literal
    would move scores and (at ties) ranks relative to the numpy control."""
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


class PgvectorAdapter(Adapter):
    name = "pgvector"
    index = "hnsw"                       # "hnsw" | "none"
    MODEL = os.environ.get("BENCH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    def __init__(self, connect=None, encoder=None):
        # Both injectable so the unit suite runs with no Postgres and no model.
        self._connect = connect
        self._encoder = encoder
        self.conn = None
        self.table = None
        self.dim = None
        self.n = 0
        self.server_version = None
        self.pgvector_version = None
        self.ef_search = None

    # -- plumbing -------------------------------------------------------
    def _encode(self, texts: list[str]):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.MODEL)
        return self._encoder.encode(texts, normalize_embeddings=True, batch_size=64,
                                    show_progress_bar=False)

    def _conn(self):
        if self.conn is None:
            if self._connect is not None:
                self.conn = self._connect()
            else:
                import psycopg  # imported late: only this adapter needs it
                self.conn = psycopg.connect(DSN, autocommit=True)
            cur = self.conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("SELECT current_setting('server_version'), "
                        "(SELECT extversion FROM pg_extension WHERE extname = 'vector')")
            row = cur.fetchone()
            if row:
                self.server_version, self.pgvector_version = row[0], row[1]
            ef = os.environ.get("BENCH_PG_EF_SEARCH")
            if ef and self.index == "hnsw":
                cur.execute(f"SET hnsw.ef_search = {int(ef)}")
        return self.conn

    def _read_ef_search(self, cur) -> None:
        # `hnsw.ef_search` only exists once the extension's library is loaded
        # in this session (a plain SHOW before the first index use raises
        # "unrecognized configuration parameter"), so it is read after the
        # first HNSW query; NULL there means the server default, 40.
        if self.index != "hnsw" or self.ef_search is not None:
            return
        cur.execute("SELECT current_setting('hnsw.ef_search', true)")
        r = cur.fetchone()
        self.ef_search = int(r[0]) if r and r[0] else 40

    # -- adapter --------------------------------------------------------
    def open(self, run_dir: str) -> None:
        # Table name from the store dir: unique per question, stable, safe.
        self.table = "bench_" + self.index + "_" + hashlib.sha1(run_dir.encode("utf-8")).hexdigest()[:16]
        self.dim = None
        self.n = 0
        cur = self._conn().cursor()
        # A stale table would silently answer another question's queries.
        cur.execute(f"DROP TABLE IF EXISTS {self.table}")

    def _create(self, dim: int) -> None:
        cur = self._conn().cursor()
        cur.execute(f"CREATE TABLE {self.table} (id bigserial PRIMARY KEY, item_id text NOT NULL, "
                    f"body text NOT NULL, emb vector({dim}) NOT NULL)")
        if self.index == "hnsw":
            cur.execute(f"CREATE INDEX ON {self.table} USING hnsw (emb vector_cosine_ops)")
        self.dim = dim

    def ingest(self, items: Iterable[MemoryItem]) -> None:
        batch = [it for it in items]
        if not batch:
            return
        emb = self._encode([it.text for it in batch])
        if self.dim is None:
            self._create(len(emb[0]))
        cur = self._conn().cursor()
        # One INSERT per turn, in history order, into the live index — the
        # agent-memory write shape. (A bulk COPY + index-after-load is faster
        # and would flatter ingest; nobody's agent writes that way.)
        for it, v in zip(batch, emb):
            cur.execute(f"INSERT INTO {self.table} (item_id, body, emb) VALUES (%s, %s, %s::vector)",
                        (it.id, it.text, vec_literal(v)))
        self.n += len(batch)

    def recall(self, query: str, k: int, when=None) -> list[RecallHit]:
        if self.dim is None or not self.n:
            return []
        q = vec_literal(self._encode([query])[0])
        cur = self._conn().cursor()
        cur.execute(f"SELECT item_id, 1 - (emb <=> %s::vector) AS score, left(body, 200) FROM {self.table} "
                    f"ORDER BY emb <=> %s::vector LIMIT %s", (q, q, int(k)))
        out = [RecallHit(id=r[0], score=float(r[1]), text=r[2] or "") for r in cur.fetchall()]
        self._read_ef_search(cur)
        return out

    def footprint_bytes(self) -> int:
        if self.dim is None:
            return 0
        cur = self._conn().cursor()
        cur.execute("SELECT pg_total_relation_size(%s)", (self.table,))
        r = cur.fetchone()
        return int(r[0]) if r and r[0] is not None else 0

    def ingest_stats(self) -> dict:
        """No LLM, nothing dropped — recorded anyway so the row says so rather
        than leaving a blank that could mean either. Numeric only: run.py sums
        these across stores."""
        return {"llm_calls": 0, "dropped_turns": 0, "ingest_errors": 0}

    def describe(self) -> dict:
        """The configuration under test, for the manifest."""
        return {"index": self.index, "ef_search": self.ef_search, "encoder": self.MODEL,
                "server_version": self.server_version, "pgvector_version": self.pgvector_version,
                "write_shape": "one INSERT per turn into a live index"}

    def close(self) -> None:
        if self.conn is not None and self.table:
            try:
                self.conn.cursor().execute(f"DROP TABLE IF EXISTS {self.table}")
            except Exception:  # noqa: BLE001 — a failed drop must not lose the row
                pass
        self.table = None


class PgvectorExactAdapter(PgvectorAdapter):
    name = "pgvector_exact"
    index = "none"
