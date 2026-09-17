# Results

Every table here comes from a run directory with a `manifest.json`; the numbers are
copied, not rounded up. Losses stay in the table.

## 2026-09-17 — LongMemEval `oracle`, 30 questions, k=5 (debain2, 20 cores)

Retrieval only (phase 1). Same 30 questions, same order, same k, one fresh store per
question. kannaka 0.16.6 (per-item CLI: **one process per remember and per recall** —
these latencies are the process start plus the encoder load, not the medium; see
"what changes next"). `oracle` haystacks contain only the evidence sessions, so
hit@k saturates for everyone; recall@k (how much of the evidence came back) is the
number that separates them here.

| adapter | n | hit@k | recall@k | MRR | recall p50 ms | p95 ms | ingest ms/item | bytes/item |
|---|---|---|---|---|---|---|---|---|
| kannaka | 30 | 1.000 | **0.909** | 1.000 | 704 | 859 | 670.7 | 314 786 |
| vector_numpy (MiniLM, exact cosine) | 30 | 1.000 | 0.870 | 1.000 | 268 | 510 | 40.2 | 1 564 |
| recency (plain-context floor) | 30 | 1.000 | 0.462 | 1.000 | 0 | 0 | 0.0 | 1 299 |

Runs: `oracle-30-kannaka-recency` (kannaka + recency), `oracle-30-vector` (vector_numpy);
dataset sha256 `821a2034d219…`.

Reading it honestly:
- kannaka brings back more of the evidence than exact-cosine MiniLM on the same
  encoder family (+0.039 recall@k), on the easiest variant. That is a small edge, on a
  small n, on a set where everyone finds *something*. It is not yet a claim.
- kannaka **loses** on latency by 2.6× at p50 and on ingest by 17×, and its store is
  200× larger per item. Part of that is real (the HRM keeps more than a vector per
  memory); most of the latency is the process-per-call shape of the 0.16.6 CLI.
- The plain-context floor is where the real tests start: on `longmemeval_s`
  (~500 items per question) recency cannot hold the evidence in a window of 5.

What changes next: kannaka-memory #973 added `remember --batch` / `recall --batch`
(one process per store); the suite uses them when the binary has them. The next
table is `longmemeval_s` with that binary — the first run that measures the medium
rather than the spawn.
