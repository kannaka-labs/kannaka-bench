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

## 2026-09-17 — LongMemEval `s`, 30 questions (5 per type), k=5 (debain2)

The first run that measures the medium: ~500 items per store, one fresh store per
question, kannaka at `e410d55` with `remember --batch` (one process, one cache
rebuild per store). Same questions, same order, same k for all three.

| adapter | n | hit@k | recall@k | MRR | recall p50 ms | p95 ms | ingest ms/item | bytes/item |
|---|---|---|---|---|---|---|---|---|
| vector_numpy (MiniLM, exact cosine) | 30 | **0.967** | **0.861** | **0.918** | 246 | 406 | 17.2 | 1 556 |
| kannaka | 30 | 0.533 | 0.444 | 0.422 | 2 661 | 4 461 | 187.0 | 42 229 |
| recency (plain-context floor) | 30 | 0.100 | 0.058 | 0.100 | 0 | 0 | 0.0 | 1 044 |

hit@k by question type (n=5 each):

| type | kannaka | vector_numpy | recency |
|---|---|---|---|
| knowledge-update | 0.800 | 1.000 | 0.200 |
| multi-session | 0.400 | 0.800 | 0.000 |
| single-session-assistant | 0.200 | 1.000 | 0.000 |
| single-session-preference | 1.000 | 1.000 | 0.000 |
| single-session-user | 0.400 | 1.000 | 0.000 |
| temporal-reasoning | 0.400 | 1.000 | 0.400 |

Run: `s-5pertype-all3`; dataset sha256 `08d8dad4be43…`.

**kannaka loses this one on every axis.** At ~500 memories per store, exact cosine
over the same encoder family finds the evidence session 97% of the time; kannaka
finds it 53% of the time, ranks it worse when it does (MRR 0.42 vs 0.92), answers
10× slower and stores 27× more bytes per item. The one type it holds is
single-session-preference (5/5 for both); it is weakest where the evidence is a
single assistant turn (1/5) and on multi-session questions (2/5).

What the misses look like (from `results.jsonl`): no facet or dream ids leak into
the top-k; kannaka's five hits often carry the *same* session two or three times
(turn-level duplicates crowding k), and on the clean misses the returned sessions
are unrelated — so this is ranking quality on a 500-item store, not a scoring
artefact. An ablation (`bench/probe.py`: facet decomposition on/off × timestamps
on/off) is running; its numbers will be appended here, and the fix work starts from
whichever setting it blames — or from the recall path itself if it blames none.

The earlier oracle-set edge (+0.039 recall@k) does not survive a real haystack.

### Follow-up on the `s` loss (same day)

- **Ablation** (`bench/probe.py`, one question per type, four configs): facet decomposition
  on/off and timestamps on/off change nothing — identical hit pattern per question. Not the cause.
- **Root cause found in the store, not the medium:** a fresh kannaka store writes
  `.encoder = hash:384:42` — a *hashing* encoder with no semantics. The table above therefore
  compares hash embeddings against MiniLM cosine. It stays, because it is what the shipped
  binary does for a new user; but it is not a measurement of the medium.
- **The fair row** is `kannaka_minilm`: the same binary with all-MiniLM-L6-v2 as its encoder —
  the *identical* weights the `vector_numpy` row uses, served to kannaka through a small
  in-process server speaking ollama's `/api/embed` (`bench/embed_server.py`), because ollama's
  own MiniLM embed cost 1.3–2.3 s per call on both lab boxes. Running now on the same 30
  questions; appended below when done.
- Two things this surfaced for kannaka-memory regardless of that row: the default encoder for
  a new store, and recall latency of 2.5–3.9 s per query in-process on a 500-item store.

### The fair row: same embeddings, medium vs exact cosine (`longmemeval_s`, same 30 questions, k=5)

`kannaka_minilm` = the kannaka binary at `2b35b46` with all-MiniLM-L6-v2 as its encoder, served
through `bench/embed_server.py` — the *identical* weights and normalisation the `vector_numpy` row
uses. Run `s-5pertype-kannaka-minilm2`.

| adapter | n | hit@k | recall@k | MRR | recall p50 ms | p95 ms | ingest ms/item | bytes/item |
|---|---|---|---|---|---|---|---|---|
| vector_numpy (MiniLM, exact cosine) | 30 | **0.967** | **0.861** | **0.918** | **246** | 406 | **17.2** | **1 556** |
| kannaka_minilm (medium + ξ reranker) | 30 | 0.933 | 0.828 | 0.911 | 2 817 | 4 169 | 381.3 | 42 229 |
| kannaka (shipped default, hash encoder) | 30 | 0.533 | 0.444 | 0.422 | 2 661 | 4 461 | 187.0 | 42 229 |
| recency | 30 | 0.100 | 0.058 | 0.100 | 0 | 0 | 0.0 | 1 044 |

hit@k by type (n=5): kannaka_minilm — knowledge-update 1.0, multi-session 0.8, single-session-
assistant 1.0, single-session-preference 1.0, single-session-user 0.8, temporal-reasoning 1.0.
vector_numpy differs only on single-session-user (1.0).

Reading it:
- **The encoder was the loss.** Given the same embeddings, the medium retrieves within 0.034 hit@k
  and 0.007 MRR of exact cosine on this set — one question in thirty. The 0.533 row is what a new
  user gets today, and that is a default-encoder decision, not a property of the medium.
- **It is not (yet) better than cosine either.** On retrieval alone, at ~500 items, the medium
  matches a numpy dot product; it does not beat it. Whatever the medium adds has to show up in the
  answer-quality phase, in consolidation over time, or at larger scale — this table does not show it.
- **Latency and footprint are the real losses:** recall 11× slower at p50 (the process-per-recall
  shape plus a store save on every recall — `resonate_query` observes and marks dirty), ingest 22×
  slower per item, 27× the bytes per item.
- **The ξ-diversity reranker is a non-factor here:** with `KANNAKA_RECALL_XI_BOOST=off` (kannaka-memory
  #975) the six probe questions came out hit-for-hit identical (5/6 either way), and the one traced
  inversion it caused (gold 5th → 8th) was a miss under both. Facet decomposition on/off with these
  embeddings is running; appended when done.

The earlier partial (0.778 at n=9) was an early-sample artefact; the full 30 is the number.
