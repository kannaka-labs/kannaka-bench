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
  inversion it caused (gold 5th → 8th) was a miss under both.
- **Facet decomposition is a non-factor too:** `KANNAKA_FACET_DECOMPOSE=0`, with and without the ξ
  reranker, came out hit-for-hit identical to the default on the six probe questions (5/6 each;
  `probe-facet`). The remaining one-in-thirty gap to cosine is the single-session-user question the
  trace dissected, not a knob.

The earlier partial (0.778 at n=9) was an early-sample artefact; the full 30 is the number.

## 2026-09-17 — Phase 2: answer accuracy (`longmemeval_s`, same 30 questions, k=5)

Answer model `agent-brain` (Claude Sonnet 4.5 via the KAX gateway) over the top-k of each finished
retrieval run; LongMemEval-style judge (same model, CORRECT / INCORRECT against the gold). Token
counts recorded per call. `bench/answer.py`; rows in `<run>/answers.jsonl` (v1) and
`answers-v2.jsonl` (v2). Plain-context baseline (`recency --full-context`) = the whole history in
the window, capped at 120k chars (~30k tokens; the first five uncapped rows were ~94k tokens each).

**v1 answer stage** (1500-char excerpts, retrieved turns only, unordered, "say I don't know"):

| adapter | n | accuracy | acc when hit | prompt tok/q | "I don't know" |
|---|---|---|---|---|---|
| vector_numpy (MiniLM cosine) | 30 | 0.47 | 0.48 | 1 035 | 8 |
| kannaka_minilm | 30 | 0.47 | 0.50 | 1 037 | 10 |
| plain context (120k cap) | 30 | 0.30 | — | 39 915 | 18 |
| kannaka (shipped default, hash) | 30 | 0.23 | 0.31 | 781 | 22 |

**The finding:** kannaka_minilm hit@5 was 0.93, yet accuracy-when-hit only 0.50 — and in 25 of the
28 hits the exact `has_answer` turn was inside the window. The loss was the answer stage, not
retrieval: 28% of turns exceed 1500 chars (p90 2 556; the model said "cut off"); assistant-type
answers live in the reply *after* the retrieved user turn; knowledge-update rows were answered with
the older value; preference rows were told to say "I don't know" against a gold that is a suggestion.

**v2 answer stage** (`988bf27`: 6 000-char excerpts, each hit + its partner turn, chronological,
"latest wins", a preference prompt; adapter-neutral):

| adapter | n | accuracy | acc when hit | prompt tok/q | "I don't know" |
|---|---|---|---|---|---|
| vector_numpy (MiniLM cosine) | 30 | **0.63** | 0.66 | 2 112 | 4 |
| kannaka_minilm | 30 | **0.63** | 0.64 | 2 105 | 5 |
| kannaka (shipped default, hash) | 30 | 0.30 | 0.44 | 2 575 | 19 |
| plain context (120k cap) | 30 | 0.27 | — | 28 879 | 17 |

By type (v2, n=5 each): kannaka_minilm — knowledge-update 3, multi-session **1**, single-session-
assistant 4, preference 5, single-session-user 4, temporal 2; vector_numpy — 3 / **0** / 4 / 5 / 5 / 2.

Reading it:
- With the same embeddings the medium and exact cosine tie on final accuracy (0.63 / 0.63). The
  gap between "shipped default" and "with a real encoder" is the whole story: 0.30 vs 0.63.
- **Multi-session is the open loss for every system** (0–1 of 5): counting across sessions needs
  more than five excerpts. Next: k=10/15 retrieval runs (also the k at which Supermemory reports
  its 95% Recall@15 — see below).
- Plain context is *worse* than top-5 retrieval at this history length even before cost: 14× the
  tokens for 0.27.
- Cost: the whole phase 2 (120 rows, twice) was ≈ $6 at Sonnet rates; the retrieval rows are well
  under a cent each — the capped baseline is 90% of the bill.

### Competitor claims to test next: Supermemory (2026-09-17)

`github.com/supermemoryai/supermemory` (MIT) claims "#1 on LongMemEval, LoCoMo and ConvoMem",
**95% Recall@15** on LongMemEval (by type: knowledge-update 99, assistant 100, user 97,
multi-session 93, temporal 91, preference 90), ~720 tokens/query, ~50 ms profiles. LLM fact
extraction + contradiction resolution + expiry over retrieval; default embedding bge-base-en-v1.5;
local binary, ollama offline. No accuracy, judge or answer model published. Not comparable to the
tables above yet (k=15 vs k=5; recall vs accuracy). Planned: a `supermemory` adapter over the local
binary, a ConvoMem loader, and k=15 for everyone — issue #1.

## 2026-09-17 — k=15 (`longmemeval_s`, same 30 questions): retrieval and answer accuracy

Run `s-5pertype-k15` (debain2, MiniLM through `bench/embed_server.py` for both retrieval adapters;
answer stage v2, `answers-v2.jsonl`, kannaka_minilm and vector_numpy only — the plain-context
baseline does not depend on k and stands at 0.27 from the k=5 pass).

| adapter | hit@5 | hit@10 | hit@15 | recall@15 | MRR | recall p50 ms | answer accuracy @15 | prompt tok/q |
|---|---|---|---|---|---|---|---|---|
| kannaka_minilm | 0.933 | 0.967 | **1.000** | 0.950 | 0.918 | 2 686 | **0.733** | 5 471 |
| vector_numpy (MiniLM cosine) | 0.967 | 1.000 | **1.000** | 0.950 | 0.921 | 14 | **0.733** | 5 422 |
| recency (last-k) | 0.100 | 0.100 | 0.167 | 0.125 | 0.106 | 0 | — | — |

Answer accuracy by type at k=15 (n=5 each), kannaka_minilm / vector_numpy: knowledge-update 0.80 /
0.80; multi-session **0.20 / 0.00**; single-session-assistant 1.00 / 1.00; preference 0.80 / 1.00;
single-session-user 0.80 / 0.80; temporal 0.80 / 0.80. "I don't know" fell to 1 of 30 for both.

Reading it:
- **Widening the window from 5 to 15 lifts final accuracy from 0.63 to 0.73 for both**, at ~2.6×
  the prompt tokens (still ~5k, versus ~29k for the capped plain-context baseline that scores 0.27).
- **Every question has a gold session in the top 15** for both retrieval adapters, all six types.
  For the record against Supermemory's published "95% Recall@15": our session-level hit@15 is 1.00
  and our recall@15 (fraction of *all* gold sessions found) is 0.95 on this 30-question stratified
  set. Whether their number is hit or recall, and on which subset, is not published, so this is a
  neighbourhood, not a ranking.
- **Multi-session is now an aggregation loss, not a retrieval one.** Those questions average 3.2
  gold sessions; recall@15 for them is 0.80 for both adapters, and the answers count "2 items"
  where the gold is 3, "one project" where the gold is 2. The missing session is the missing
  count. Next: per-type k (or a second retrieval pass conditioned on the first) — and it is the
  first place where a memory that *consolidates* across sessions could beat cosine on this set.
- The medium and exact cosine are still tied on everything that matters to a user; the medium's
  cost is 190× recall latency and 27× bytes (kannaka-memory #977).
