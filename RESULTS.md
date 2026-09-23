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

### Session cap at k=15 (`--session-cap N`, run.py `ed3c019`; cosine first, same 30 questions)

At most N turns per session in the top-15, backfilled from a k×3 candidate list. Adapter-neutral,
no per-type tuning.

| cosine, k=15 | recall@15 | MRR | multi-session recall | temporal recall |
|---|---|---|---|---|
| no cap | 0.950 | 0.921 | 0.80 | 0.90 |
| cap 3 | 0.950 | 0.921 | 0.80 | 0.90 |
| cap 2 | 0.958 | 0.922 | 0.85 | 0.90 |
| **cap 1 (15 distinct sessions)** | **0.983** | **0.929** | **0.95** | **0.95** |

hit@15 stays 1.000 throughout; knowledge-update stays 1.0. One session per slot is the right
shape for LongMemEval's session-level questions: the answer stage already expands each hit to
its turn pair, so nothing is lost by taking one turn per session. kannaka_minilm at cap 1 and the
answer pass for both are running; appended when done.

**The cap-1 answer pass is a loss (cosine, k=15, cap 1, answer stage v2):** accuracy **0.500**,
down from 0.733 without the cap; by type knowledge-update 0.80, multi-session 0.20,
single-session-assistant **0.20 (was 1.00)**, preference 0.60, user 0.60, temporal 0.60; prompt
~6.2k tokens. Retrieval found more sessions and threw away the turn that carried the answer: the
answer-bearing turn is often *not* its session's top-ranked turn, and one-turn-per-session keeps
only the top-ranked one. Multi-session did not gain either (0.20): the counts need the specific
turns, not just the sessions. So session diversity and within-session coverage trade against each
other at a fixed budget, and the cap alone is the wrong knob. Next: keep the candidate list per
row and let the answer stage take the top-M turns of each selected session (`--per-session`),
measured offline on the same retrieval rows.

**Answer-stage shapes on the same cosine k=15 rows** (`--per-session M` takes the top-M turns of
each selected session from the k×3 candidate list; every row pair-expanded, chronological):

| shape (cosine, k=15) | excerpts/q | prompt tok/q | accuracy | multi-session | single-session-assistant |
|---|---|---|---|---|---|
| **no cap, top-15 turns** | ~30 | **5 422** | **0.733** | 0/5 | 5/5 |
| cap 1, 1 turn/session | ~30 | 6 240 | 0.500 | 1/5 | 1/5 |
| cap 1, 2 turns/session | 35 | 8 753 | 0.633 | **2/5** | 4/5 |
| cap 1, 3 turns/session | 43 | 10 962 | 0.567 | 2/5 | 3/5 |

Verdict: **the plain ranked top-15 turns is the best shape at every budget tried**; more excerpts
cost accuracy, not just tokens. Session diversity buys two multi-session answers and loses more
elsewhere. k=15, no cap, turn pairs stays the standard setting. Multi-session (counting across
3+ sessions) remains the open loss for every system on this set — an aggregation problem the
answer stage cannot fix by widening, which is the case for a memory that consolidates.

## 2026-09-17 — Supermemory, head to head (`longmemeval_s`, same 30 questions, k=15)

`github.com/supermemoryai/supermemory` (MIT), self-hosted `supermemory-server` on debain2, adapter
`bench/adapters/supermemory.py`, **retrieval-only row**: `taskType=superrag` ingest, `searchMode=documents`
search, embeddings through the same MiniLM weights as every other row (their server → our embed
server's OpenAI-shaped endpoint). Their memory engine (`supermemory_mem`: LLM fact extraction,
`memories` search) is not in this table — it needs a real LLM per document and is run and costed
separately. Answer stage v2, standard setting. Run `s-5pertype-k15-supermemory2`.

| adapter (k=15) | hit@15 | recall@15 | MRR | recall p50 ms | ingest ms/item | answer accuracy | prompt tok/q |
|---|---|---|---|---|---|---|---|
| kannaka_minilm (medium) | **1.000** | **0.950** | 0.918 | 2 686 | 370 | **0.733** | 5 471 |
| vector_numpy (MiniLM cosine) | **1.000** | **0.950** | **0.921** | **14** | **14** | **0.733** | 5 422 |
| supermemory (documents mode) | 0.897 | 0.822 | 0.874 | 134 | 786 | 0.621 | 3 193 |
| recency | 0.167 | 0.125 | 0.106 | 0 | 0 | — | — |

Supermemory by type (hit / recall / accuracy, n=5 unless noted): knowledge-update 0.80 / 0.70 / 0.40;
multi-session 0.80 / 0.68 / 0.20; single-session-assistant 1.00 / 1.00 / 1.00 (n=4); preference
0.80 / 0.80 / 0.80; single-session-user 1.00 / 1.00 / 0.80; temporal 1.00 / 0.78 / 0.60.
n=29 of 30: question `4c36ccef` (single-session-assistant) is an **error row** — its store never finished
ingesting (35 of 521 documents stuck in the server's queue after three fresh restarts and a 15-minute
drain), so it is neither a hit nor a miss; the table is over the 29 that completed.

Reading it:
- On this set, at the same k and the same embeddings, **their chunk retrieval finds the gold session
  less often than plain cosine over turns (0.897 vs 1.000)** and the answer model is right less often
  (0.621 vs 0.733). Their headline "95% Recall@15" is not reproduced here for the retrieval-only path;
  whether their number comes from the extraction layer, a different embedding (bge-base by default),
  or a different subset is not published.
- Their search is fast (134 ms p50, 20× faster than kannaka's spawn-per-recall path) and their
  prompts are smaller (3.2k tokens: chunks, not turn pairs); their ingest is the slowest of the
  table (786 ms/item through their async pipeline).
- **Operational findings, all reproducible:** the server calls an LLM for every document even in
  chunk-only mode (`bench/llm_stub.py` answers those instantly for this row); "Self-hosted lite is
  licensed for up to 10,000 documents" (HTTP 403 after 19 stores — the adapter now deletes a store
  on close); its ingest queue paused at the default 1 GB ingest-memory limit (raised to 8 GB); and it
  slowed from ~3 to ~30 min per 500-turn store over the run with internal KV-persistence errors,
  needing two fresh restarts. None of that is a knock on the algorithm; all of it is what a
  self-hoster meets.

## 2026-09-18 — Scaling: kannaka ingest and recall cost vs store size (kannaka-memory #978)

Fourteen ConvoMem context-20 stores, `kannaka_minilm` with `remember --batch` (bulk mode, #974) and
the in-process MiniLM embed server (74 ms/call, so embedding is not the cost), debain2:

| items in store | ingest ms/item | whole-store ingest | recall p50 ms |
|---|---|---|---|
| 241 | 136 | 33 s | 1 317 |
| 436 | 241 | 105 s | 2 657 |
| 652 | 398 | 4.3 min | 3 655 |
| 811 | 533 | 7.2 min | 5 867 |
| 958 | 695 | 11.1 min | 11 488 |
| 5 192 | — | **not finished after 2 h 8 min** (killed) | — |

Least squares over the fourteen: **ingest ms/item ≈ −74 + 0.77·n**, i.e. the cost of one insert
grows linearly with what is already in the store, so a store costs O(n²) — at 5 000 items that is
~5.5 h; **recall ms ≈ −2 060 + 11.7·n** (the per-recall save path, #977, on top of the spawn). Cosine
over the same items: 3–17 ms/item ingest and 15 ms recall, flat. This is the loss that keeps
LongMemEval-M and ConvoMem beyond context ~20 off the table today; the kannaka rows on ConvoMem
below are run with `--max-items 1500`, and every skipped store is an explicit row.

## 2026-09-18 — ConvoMem (Salesforce, CC-BY-NC-4.0), context 20, five questions per category, k=15

`--dataset convomem --convomem-context 20 --limit 5 --k 15`; session-level gold = the evidence
conversation ids; abstention questions carry no gold and are unscored (the answer stage will score
the refusal). Runs `convomem-c20-5percat-vector` and `convomem-c20-5percat-kannaka` (the latter
with `--max-items 1500`: three assistant-facts stores of 5 192 turns each are explicit skip rows,
kannaka-memory #978). Comparison over the 22 questions both adapters scored:

| adapter (k=15) | hit@15 | recall@15 | MRR | recall p50 ms | ingest ms/item |
|---|---|---|---|---|---|
| kannaka_minilm (medium) | **1.000** | **0.871** | 0.865 | 7 024 | 455 |
| vector_numpy (MiniLM cosine) | **1.000** | 0.848 | 0.865 | **15** | **3** |
| recency (all 25 scored) | 0.200 | 0.053 | 0.166 | 0 | 0 |

recall@15 by type (kannaka / cosine): assistant_facts 0.92 / 0.92 (n=2), changing 0.93 / 0.93,
implicit_connection 0.73 / 0.73, preference **1.00 / 0.90**, user 0.80 / 0.80.

Reading it:
- Every scored question has its gold conversation in the top 15 for both. On the multi-evidence
  categories the medium recovers one more gold conversation than exact cosine (preference), the
  first place in any table where it edges cosine on the same embeddings — one question in
  twenty-two, so a hint, not a result.
- The cost side is unchanged: 470× the recall latency and 150× the ingest cost per item, and three
  stores it could not ingest at all inside the cap. The scaling section above is the reason.
- Next on ConvoMem: the answer pass at the standard setting (accuracy plus how the abstention
  questions are refused), and larger contexts once #978 lands.

**ConvoMem c20 answer accuracy** (answer stage v2 standard setting; `answers-v3` = 600-token answer
cap, `answers-v2` = the old 200; the cap turned out not to be the loss — outputs averaged 150–190
tokens either way):

| adapter | n | accuracy (v2 → v3) | abstention | assistant_facts | changing | implicit | preference | user | prompt tok/q |
|---|---|---|---|---|---|---|---|---|---|
| vector_numpy | 30 | 0.567 → 0.533 | 5/5 | 0/5 | 2/5 | 4/5 → 3/5 | 4/5 | 2/5 | 1 196 |
| kannaka_minilm | 27 (+3 skipped) | 0.593 → 0.630 | 5/5 | 0/2 | 2/5 | 3/5 → 4/5 | 4/5 | 2/5 | 1 229 |

The one-question swings between v2 and v3 are judge noise on the same excerpts (temperature 0,
different answer length). Reading it:
- **Both systems refuse every abstention question correctly** (5/5, "I don't know" with the reason).
- **The loss is multi-evidence coverage, not the cap and not the store.** ConvoMem's user /
  changing / assistant_facts gold answers are lists assembled from 2–6 evidence turns spread over
  several conversations ("six key items…", six books' prices, three meetings). Session-level
  retrieval finds *a* gold conversation every time, but the top-15 turns hold only some of the
  evidence turns, the model answers with the part it has ("I can identify four of the six…"), and
  the strict judge marks the partial list wrong — including one answer whose computed averages
  matched the gold exactly but listed two misses instead of three. Same shape as LongMemEval's
  multi-session loss, in a dataset built around it.
- Next: a turn-level "all evidence turns covered" column (the diagnostic already shows *any*
  evidence turn present in 4–5 of 5), and the aggregation story that a consolidating memory has to
  tell here.

## 2026-09-18 — Evidence coverage: are the turns the answer is built from in the top-k?

`evid@k` (run.py `1cb0c2e`; `python -m bench.evidence_coverage <run>` for older runs): of a
question's answer-bearing turns inside its gold sessions (`has_answer`), the fraction found in the
top-k; `all` = questions where every such turn is there. Session-level hit@k answers "is a gold
conversation present"; this answers "is the evidence present". k=15 throughout.

| run | adapter | evidence turns / q | evid@15 | all-evidence@15 |
|---|---|---|---|---|
| longmemeval_s (30 q) | **kannaka_minilm** | 2.0 | **0.848** | **0.767** |
| longmemeval_s | vector_numpy | 2.0 | 0.809 | 0.700 |
| longmemeval_s | supermemory (documents) | 2.0 | 0.657 | 0.517 |
| longmemeval_s | recency | 2.0 | 0.097 | 0.067 |
| ConvoMem c20 (23 q with matched evidence) | kannaka_minilm (20 q) | 3.9 | 0.443 | 0.150 |
| ConvoMem c20 | vector_numpy | 4.1 | 0.400 | 0.130 |

By type, longmemeval_s, kannaka / cosine (coverage): multi-session **0.59 / 0.49**, preference
**0.80 / 0.67**, single-session-user 0.90 / 0.90, temporal 0.80 / 0.80, knowledge-update and
single-session-assistant 1.00 / 1.00.

Reading it:
- **At the turn level the medium is ahead of exact cosine on the same embeddings**: +0.04 coverage
  overall, +0.10 on multi-session and +0.13 on preference, with all-evidence 0.767 vs 0.700. This
  is the first table where the medium's ranking, not just its session hit, beats the dot product —
  and it is exactly the multi-evidence questions. Final accuracy still ties (0.733 both) because the
  answer model does not always use what it is given; the retrieval edge is real and small.
- **Supermemory's chunk retrieval is well behind on evidence (0.657 / 0.517)**: its chunks land in
  the right conversation more often than they land on the right turn.
- **ConvoMem is the hard case in numbers:** four evidence turns per question, and the top-15 holds
  all of them for only 13–15 % of questions. That, not the answer stage, is why user / changing /
  assistant_facts accuracy sits at 0–2 of 5. Aggregation needs either a much wider window or a
  memory that has already joined the evidence before the question arrives.

## 2026-09-18 — Encoder row: bge-base-en-v1.5 (`longmemeval_s`, same 30 questions, k=15)

`vector_bge` and `kannaka_bge` (`a2d3454`) over BAAI/bge-base-en-v1.5 (768-d, Supermemory's
default encoder) through the bench embed server (126 ms/call on debain2), same weights for both;
answer stage v2 standard setting. Run `s-5pertype-k15-bge`.

| encoder → adapter | hit@15 | recall@15 | evid@15 | MRR | recall p50 ms | ingest ms/item | accuracy | prompt tok/q |
|---|---|---|---|---|---|---|---|---|
| MiniLM → kannaka_minilm | 1.000 | 0.950 | 0.848 | 0.918 | 2 686 | 370 | 0.733 | 5 471 |
| MiniLM → vector_numpy | 1.000 | 0.950 | 0.809 | 0.921 | 14 | 14 | 0.733 | 5 422 |
| **bge-base → kannaka_bge** | 0.967 | 0.942 | 0.816 | 0.950 | 3 063 | 792 | **0.833** | 5 886 |
| **bge-base → vector_bge** | **1.000** | **0.950** | 0.816 | **0.952** | **31** | 99 | 0.767 | 5 926 |
| MiniLM → supermemory (documents) | 0.897 | 0.822 | 0.657 | 0.874 | 134 | 786 | 0.621 | 3 193 |

Accuracy by type (bge, kannaka / cosine, n=5): knowledge-update 4 / 4, multi-session **3 / 2**,
single-session-assistant 5 / 5, preference **5 / 4**, single-session-user 4 / 4, temporal 4 / 4.

Reading it:
- **The stronger encoder lifts final accuracy for both** (0.733 → 0.767 cosine, 0.733 → 0.833
  medium) and MRR (0.92 → 0.95). Retrieval at the session level is saturated on this set (hit@15
  1.000 / 0.967); the gains are in rank and in what the answer model gets.
- **0.833 is the best accuracy in any table here**, on the medium; the two questions it wins over
  cosine are one multi-session and one preference, with *identical* evidence coverage (0.816 for
  both) — so this is rank order inside the window plus answer-model variance, not more evidence.
  Two questions in thirty is inside the noise of a 30-question set; it is reported, not claimed.
- The medium's turn-level edge from the MiniLM table (0.848 vs 0.809) does not reappear with
  bge-base (0.816 / 0.816): whatever the medium adds to a weak encoder's ranking, a strong encoder
  already has.
- The one session miss for kannaka_bge (multi-session, 0.80) is the trade for the same medium
  behaviour seen on MiniLM; cost columns are as before (100× recall latency, 8× ingest).
- Supermemory's default encoder does not explain its retrieval gap: on the same weights the
  chunk index is at 0.897 / 0.822 while both turn-level systems are at ≥ 0.967 / 0.942.

## 2026-09-18 — bge-base at n=60 (`longmemeval_s`, ten questions per type, k=15)

Run `s-10pertype-k15-bge` (seeded with the 30-question rows via `--resume`; the 30 new questions
ingested fresh). Same setting as the 30-question bge table.

| adapter (bge-base, k=15) | n | hit@15 | recall@15 | evid@15 | all-evidence | MRR | recall p50 ms | accuracy |
|---|---|---|---|---|---|---|---|---|
| kannaka_bge | 60 | 0.983 | 0.960 | **0.859** | **0.783** | 0.939 | 3 081 | **0.800** |
| vector_bge | 60 | **1.000** | **0.969** | 0.852 | 0.767 | **0.940** | **32** | 0.783 |

Accuracy by type (kannaka / cosine, n=10): knowledge-update 9 / 9, multi-session **5 / 4**,
single-session-assistant 10 / 10, preference **9 / 8**, single-session-user **8 / 9**, temporal 7 / 7.

Reading it:
- **Doubling n shrank the accuracy gap from +0.067 to +0.017** (one question in sixty): the
  30-question edge was mostly noise, as flagged. The honest summary of the medium against exact
  cosine on the same strong encoder is **parity on accuracy (0.80 vs 0.78), a whisker ahead on
  turn-level evidence (0.859 vs 0.852), a whisker behind on session hit (0.983 vs 1.000), and 100×
  the recall latency**.
- **Multi-session is the open loss for both at n=10** (5 / 4 of 10) with all-evidence 0.50: half
  of those questions do not have all their evidence turns in the top-15 for either system. Widening
  k does not fix it (the cap ablation above); consolidation is the untested route.
- Retrieval at the session level is saturated on LongMemEval-S; the paper's LongMemEval story is
  the encoder (0.53 → 0.93 → 0.98) and the answer stage (0.47 → 0.63 → 0.73 → 0.80), not the index.

## 2026-09-19 — The consolidation arm: does dreaming help? (`longmemeval_s`, 30 q, bge-base, k=15)

The experiment the paper's central claim needs (`docs/paper-results-skeleton.md` §5): ingest the
haystack, run **one deep dream cycle** (`kannaka dream --mode deep`, `run.py --consolidate`,
`9743018`), then recall. Everything else identical to the bge row above. Run
`s-5pertype-k15-bge-dream`.

| kannaka_bge, k=15 | hit@15 | recall@15 | evid@15 | all-evidence | MRR | accuracy | recall p50 ms | ingest ms/item | bytes/item |
|---|---|---|---|---|---|---|---|---|---|
| no consolidation | 0.967 | 0.942 | 0.816 | 0.733 | **0.950** | 0.833 | 3 068 | **792** | **42 229** |
| **+ one deep dream** | 0.967 | 0.942 | 0.816 | 0.733 | **0.529** | 0.833 | **1 344** | 1 109 | 55 624 |

Dream cost: **141 s per store** (p50, ~500 turns), 9 memories minted per store, +32 % bytes.

Reading it — this is a negative result, stated plainly:
- **Consolidation changed nothing measurable about what is retrieved.** hit@15, recall@15, evidence
  coverage, all-evidence and final accuracy are *identical* to the same run without it, question for
  question (0.833 both; multi-session 4/5 vs 3/5 is one question).
- **It damaged the ranking.** MRR fell 0.950 → 0.529 because a synthesized memory took **rank 1 in
  24 of 30 questions** (48 dream rows inside the top-15 overall, 1.6 per question). Dreams cannot
  be evidence — they are cross-cluster syntheses, not turns — so every one of them is a slot taken
  from a turn that could be. The answer stage survived it here only because k=15 is wide enough to
  carry the evidence anyway; at k=5 this would be a direct accuracy loss.
- This is the **dream-attractor pathology already quantified on the live store** (kannaka-memory
  #963: 183 dream rows held 75 % of the Gram mass and answered 4 of 5 unrelated probes). The bench
  now reproduces it on a clean corpus in one cycle, with a number attached.
- One unexplained win: **recall got 2.3× faster after dreaming** (3 068 → 1 344 ms p50). Worth a
  look — if consolidation shrinks the working set that is the first cost-side result in its favour.

What this does to the claim: *on this benchmark, one dream cycle does not join evidence for
multi-evidence questions, and it costs 141 s, a third more bytes, and the top of the ranking.* The
architecture's distinguishing mechanism is measured and, as it stands today, it does not pay for
itself. The honest next questions are whether repeated cycles over a longer-lived store behave
differently (every store here is one-shot), and whether dream rows should be excluded from recall
by default (kannaka-memory #963 / a new issue).

## 2026-09-20 — answer stage v4: enumerate-then-compute, and two instrument fixes

Same retrieval rows as the k=15 run above (`s-5pertype-k15`, nothing re-retrieved), re-answered
with `bench/answer.py --tag v4`. Three changes, each measurable on its own.

**1. An aggregation route.** Multi-session scored 0.20 / 0.00 at k=15 while hit@15 was 1.000 — every
gold session was already in the window. Reading the four kannaka failures, none is truncation
(86–185 completion tokens against a 600 cap) and none is a retrieval miss: the model states the
components and stops short of the arithmetic. Gold `3.5 weeks` came back as "**two weeks**" and "**a
week and a half**" as two bullets, never added. Gold `5 model kits` was answered `4`; gold `3 items`
was answered `2`; gold `2 projects` found the second project and then reasoned it away. So the
aggregate prompt asks for two explicit steps — enumerate every candidate with its date, sweeping all
excerpts and dropping nothing on the model's own reasoning; then compute, adding components for a
combined total rather than naming them separately.

The prompt is **additive, not a replacement**: it keeps the chronology and latest-wins rules
verbatim. That matters because surface form cannot separate the types — "How many Korean restaurants
have I tried?" (knowledge-update, latest-wins) is the same shape as "How many model kits have I
worked on?" (multi-session, enumerate-all), and 4 of 5 temporal questions also open "How many
days…". A replacement prompt routed onto those would put two 0.80 categories at risk.

**2. Routing reads the question, not the gold label.** v2 chose its prompt with
`"preference" in r["qtype"]` — the dataset's own type label, at inference time. No deployed system
has that, so the published 0.733 was optimistic by whatever the preference prompt is worth.
`--route question` (now the default) reads the question text; `--route qtype` reproduces the old
behaviour. On these 30 questions the honest router reproduces the gold-label routing **exactly, 0
disagreements** — so the leak cost nothing here, but it should not have been in the number's path.
Routing by gold type: multi-session 5/5 → aggregate, preference 5/5 → preference,
single-session-assistant 5/5 → plain, plus 7 rows of other types that ask for a number.

One trap worth recording: matching a bare "recommend" sends *"Can you remind me … the restaurant you
recommended"* (single-session-assistant, scoring **1.00**) to the preference prompt, which forbids
"I don't know" and asks for a generic answer — replacing correct recall with invention. Past-tense
lookups are excluded explicitly.

**3. A deterministic abstention guard on the judge.** `JUDGE_SYS` says an "I don't know" answer is
INCORRECT unless the reference says the information is unavailable. The judge does not reliably
comply: on one temporal question it graded two near-identical abstentions **differently** for the two
adapters, handing vector_numpy a free point for an answer ending "I don't know" against gold `2`. A
judge that lets abstentions through rewards whichever system answers least, so the rule is now
enforced in code (`grade()`), recorded per row as `judge_overridden`. Replayed over all 120 already-
graded rows of this run it flips that one row and **nothing else** — v2 is unaffected, so the 0.733
baseline above stands as published.

| adapter | v2 | v4 raw | **v4 corrected** | multi-session v2 → v4 |
|---|---|---|---|---|
| kannaka_minilm | 0.733 | 0.800 | **0.800** | 0.20 → **0.40** |
| vector_numpy | 0.733 | 0.767 | **0.733** | 0.00 → **0.20** |

By type (v4 corrected, n=5 each), kannaka_minilm / vector_numpy: knowledge-update 0.80 / 0.80,
multi-session **0.40 / 0.20**, single-session-assistant 1.00 / 1.00, preference 1.00 / 0.80,
single-session-user 0.80 / 0.80, temporal-reasoning 0.80 / 0.80. Prompt ~5.5k tok/q, essentially
unchanged from v2's 5.4k — this buys nothing with context.

Reading it, conservatively:
- **The diagnosed mechanism is fixed, and that is the solid part.** The `3.5 weeks` row — components
  found but never summed — is now answered `3.5 weeks` by **both** adapters. That is the specific
  failure the change was built for, confirmed twice.
- **The aggregate totals are not yet evidence.** n=5 per type: one question is ±0.20. kannaka gains
  two questions overall, vector_numpy none once the mis-grade is removed. The preference column moves
  +0.20 for one adapter and −0.20 for the other on *identical* routing and prompt, which is the
  signature of judge/answer nondeterminism, not of the change.
- **Nothing regressed where the risk was.** single-session-assistant held 1.00 / 1.00, and
  knowledge-update held 0.80 / 0.80 despite two of its rows taking the aggregate route — the
  evidence that keeping the prompt additive was the right call.
- **Multi-session remains the open loss** (0.40 / 0.20). Of the three still wrong for kannaka, the
  failures are now under-enumeration rather than missing arithmetic: the sweep finds most items and
  misses one. That is a different problem from the one this change fixed.

## 2026-09-21 — temporal recall: a negative result, and the bug that made it unmeasurable

`KANNAKA_RECALL_TEMPORAL_EXP` discounts a memory by how long ago it was confirmed, so a fact that
has been superseded ranks below the one that replaced it (ADR-0051 / L8). It has shipped off. This
is the first measurement of turning it on — and the first that *could* be one.

**Why no earlier run could have measured it.** `temporal_weight` decays `0.5^(age/half_life)` from a
memory's `observed_at` to *now*, then clamps the result **up** to the superseded floor (0.25). That
clamp binds at exactly two half-lives — 360 days at the default. Every corpus here is dated 2023 and
read in 2026, so **every candidate returned the floor**: the factor became a constant multiplier,
which cannot reorder anything. It does not fail and nothing reports it. Measured through the CLI on
two claims observed three months apart:

| | scores |
|---|---|
| flag off | 1.0 / 1.0 |
| flag on, wall clock | **0.25 / 0.25** — identical, ranks nothing |
| flag on, `--at` the question date | 0.996 / 0.707 — `0.5^(90/180)`, exactly |

Lowering `KANNAKA_RECALL_TEMPORAL_FLOOR` does not help; it only moves the constant (0.05 for
everything). The fix was `kannaka recall --at` (kannaka-memory#994) plus passing each question's
`asked_at` from the harness, which `run.py` had always supplied and the adapter had always dropped.

**Three arms, same 30 questions, same stores, k=15, kannaka_minilm** (`temporal2-*`):

| arm | hit@15 | recall@15 | evid@15 | MRR |
|---|---|---|---|---|
| A — flag off | 1.000 | 0.950 | 0.848 | **0.9183** |
| B — exp=1.0, half-life 180d (**the shipped setting**) | 1.000 | 0.950 | 0.848 | **0.9028** |
| C — exp=1.0, half-life 3000d | 1.000 | 0.950 | 0.848 | **0.9186** |

Retrieval quality is **identical across all three**. The flag finds nothing new, loses nothing, and
only reorders — barely: **2 of 30 questions changed at all.**

MRR by question type (n=5 each):

| qtype | A off | B hl180 | C hl3000 |
|---|---|---|---|
| knowledge-update | 1.000 | **0.900** | 1.000 |
| multi-session | 0.818 | 0.825 | 0.820 |
| single-session-assistant | 1.000 | 1.000 | 1.000 |
| single-session-preference | 1.000 | 1.000 | 1.000 |
| single-session-user | 0.692 | 0.692 | 0.692 |
| temporal-reasoning | 1.000 | 1.000 | 1.000 |

The two questions that moved, A → B:

```
knowledge-update    1.000 -> 0.500   WORSE
multi-session       0.091 -> 0.125   better
```

Reading it:
- **The one category the feature exists for is the one it hurt.** Knowledge-update questions are
  precisely the superseded-fact case; on one of those five, enabling the flag pushed the gold answer
  from rank 1 to rank 2. That is a single question, so it is not evidence that temporal ranking
  *harms* knowledge-update — but it is certainly not evidence it helps, and it is the opposite of
  the designed effect.
- **Arm C is the consistency check, not a candidate.** A 3000-day half-life makes every weight ≈1.0
  once ages are days rather than years, so C should be indistinguishable from A — and is (0.9186 vs
  0.9183). Together with B differing under deterministic scoring on identical stores, that is what
  proves `--at` actually reached the scorer.
- **Recommendation: leave the ADR-0051 Phase 3 gate closed.** On this benchmark the best case is
  neutral and the shipped setting is marginally negative. The M9/M8/M3 preconditions
  (kannaka-memory#992) stay worth having regardless — each is correct whether the flag is on or off.
- What this does NOT settle: longmemeval_s has five knowledge-update questions and no question whose
  gold answer is explicitly stamped `--supersedes`. A corpus built to exercise supersession directly
  would be a fairer test of the mechanism than one where only 2 of 30 rankings can move at all.

**A measurement fault this exposed:** the three arms differed only by environment variable, and
*no artifact recorded which arm ran with what*. The per-question store — and the `queries.ndjson`
that would have shown the per-row `at` — is deleted at the end of a run, so afterwards the arms were
distinguishable only by trusting the shell history that launched them. `run.py` now records the
`KANNAKA_RECALL_*` flags in `manifest.json`, the same way the beam trace is recorded: an arm that
cannot show what it ran with cannot be compared against one that ran differently.

## 2026-09-22 — The store shape was wrong: re-take on the corrected ingest (kannaka-memory #1039)

Every kannaka row above was measured on stores built by one `remember --batch` per question,
and kannaka-memory #917/#1031 showed that a single-process bulk load stayed on the FLAT (v1)
medium: it never reloaded, so it never became chiral, so it minted **no facet rows** even with
`KANNAKA_FACET_DECOMPOSE=1` — while the same input through separate `remember` calls produced
a chiral store with 3–4.5× the rows. Nick ruled that an ingest correctness bug, not a benchmark
artifact; #1039 (v0.16.11) makes a fresh store chiral from birth. Same 30 questions, k=15,
same MiniLM embeddings, same host; run `s-5pertype-k15-chiral`, binary built from #1039's first
commit (`3cc037c`; the second commit only syncs an in-process view that recall does not read).

| run | hit@15 | recall@15 | evid@15 | MRR | recall p50 | ingest ms/item | bytes/item | avg store |
|---|---|---|---|---|---|---|---|---|
| `postflip-k15` (flat, no facets — the published row) | 1.000 | 0.950 | 0.848 | 0.918 | 2 838 ms | 376 | 42 KB | 21 MB |
| `s-5pertype-k15-chiral` (chiral from birth, facets minted) | 0.967 | 0.958 | 0.804 | 0.919 | 4 036 ms | 255 | 185 KB | 92 MB |

Six of thirty questions moved (two better, four worse):

```
e47becba      single-session-user        evid 0.50 -> 1.00
b46e15ed      temporal-reasoning         evid 0.00 -> 0.20
b5ef892d      multi-session              evid 1.00 -> 0.50
75832dbd      single-session-preference  evid 1.00 -> 0.00
gpt4_59149c77 temporal-reasoning         evid 1.00 -> 0.50
6d550036      multi-session              hit  1    -> 0      (session-level miss)
```

Reading it, and this is a loss to publish:
- **Retrieval did not improve.** Session-level recall is within a question (0.950 → 0.958);
  evidence coverage fell 0.044, about 1.3 questions of turn-level coverage, and one
  multi-session question lost its session hit outright. With n=5 per type this is noise-level
  either way — but the facet rows, which #925 credited with a third of recall, bought nothing
  measurable here.
- **The costs are not noise.** Recall p50 is up 42% (more rows to resonate through) and the
  store is **4.4× larger on disk** (185 KB per source item; 92 MB per 500-turn store). Ingest
  per item got *faster* (376 → 255 ms) because the chiral path no longer pays the flat medium's
  O(n) interference per insert — the #1031 finding, confirmed at this n.
- So the honest published row for the standard setting is now **0.967 / 0.958 / 0.804**, not
  1.000 / 0.950 / 0.848, and the footprint column is 185 KB, not 42 KB. Every row above that
  says `kannaka_minilm` at k=15 was taken on the flat shape.

Running next: the same 30 questions on the chiral path with `KANNAKA_FACET_DECOMPOSE=0`
(`s-5pertype-k15-chiral-nofacets`), which separates "chiral from birth" from "facets" — if it
lands near the old row at the old footprint, facets are the entire cost and none of the gain
on this benchmark.

## 2026-09-22 — Laya as a System-1 reflex: two pre-registered calibrations, both losses

Laya (`NandhaKishorM/laya` 0.3.6, Apache-2) answers typed questions — `choice`/`score`/`noul`
— in one non-autoregressive forward pass, calibrated by RL against proper scoring rules. The
proposal (a "Reflex organ" that weights, routes or gates but never generates) is worth
nothing until the off-the-shelf checkpoint is calibrated on a case whose answer we know.
Decision rules were fixed before the run in `experiments/laya_reflex/README.md`; raw outputs
in `experiments/laya_reflex/results/`. Host: debain2 CPU (20 cores), the bench running alongside.

**E-L1 evidence gate.** For the 30 standard questions, one `noul` per top-15 candidate the
medium actually retrieved (450 decisions): *does the excerpt contain information needed to
answer the question?* Ground truth: LongMemEval turn-level `has_answer` (43 positives, 9.6%).

| AUROC | Brier | Brier of "always no" | ECE | precision @0.5 | recall @0.5 | latency median / p95 |
|---|---|---|---|---|---|---|
| **0.750** | 0.092 | 0.096 | 0.057 | 0.25 | 0.19 | 507 ms / 3.2 s |

Rule was AUROC ≥ 0.85 and Brier ≤ 0.15 to become a gate, 0.75–0.85 to fine-tune first, below
0.75 a loss. It landed at 0.7496 — a loss by the letter, and by the substance: Brier is a
wash against predicting "no" for everything, and at p ≥ 0.5 it finds 8 of 43 evidence turns
while flagging 24 non-evidence ones. The ranking has some signal (AUROC 0.75) but the
probabilities do not, on this domain.

**E-L2 question routing from the question text alone** (all 500 questions, `choice` over
the six types; rule: ≥ 0.80 to replace the gold-label routing in the answer stage):
**accuracy 0.314** (chance 0.167). It answers `single-session-user` for 348 of 500 questions —
every type's plurality prediction is the first criterion listed. `single-session-assistant`
0/56, `knowledge-update` 4/78. A loss; the answer stage keeps its leak-free un-routed baseline.

**Latency, the honest System-1 number:** 267 ms median for a question-only state on a free
20-core box, 507 ms with a 512-token excerpt, 2.4 s while the bench shared the box. The
README's 33 ms is a T4 figure. On CPU this is a dream-time or membrane-time reflex, not a
per-hit recall gate.

**What survives:** the fine-tuning path. LongMemEval gives ~7 500 turn-level evidence labels
on the 470 questions outside the standard 30, and Laya ships the typed-decisions fine-tune
notebook; that run needs a GPU (qBraid) and is the next Laya experiment, E-L1b, with the same
decision rule on the held-out 30. Nothing here touches a store.

## 2026-09-22 — Facets off on the chiral path: the loss was facets, all of it

Third arm, same 30 questions, same binary, `KANNAKA_FACET_DECOMPOSE=0`
(`s-5pertype-k15-chiral-nofacets`):

| run | hit@15 | recall@15 | evid@15 | MRR | recall p50 | p95 | ingest ms/item | bytes/item |
|---|---|---|---|---|---|---|---|---|
| `postflip-k15` — flat (the old published row) | 1.000 | 0.950 | 0.848 | 0.918 | 2 838 ms | 4 304 | 376 | 42 KB |
| chiral, facets on | 0.967 | 0.958 | 0.804 | 0.919 | 4 036 ms | 4 823 | 255 | 185 KB |
| **chiral, facets off** | **1.000** | **0.950** | **0.848** | **0.918** | **1 024 ms** | **1 146** | **98** | **46 KB** |

The facets-off chiral row reproduces the old flat row's retrieval **exactly** — every one of
the 30 evidence-coverage values is identical — at **2.8× lower recall latency**, **3.8× lower
ingest cost** and the old footprint. So on this benchmark:

- **Chiral-from-birth is a pure win**: same ranking as the flat medium, much cheaper on both
  paths (the flat medium paid O(n) interference per insert and converted itself on every
  load). This is the row the standard setting should carry from now on.
- **Facets are the entire cost and none of the gain**: +4.5 s p50 → 4.0 s, 4× disk, and
  retrieval within noise of no facets, slightly down. #925's "facets are a third of recall"
  was measured on a different store and corpus; here the parent rows already carry the
  turn-level evidence, so minted fragments only add rows to resonate through.
- The kannaka-memory binary defaults `KANNAKA_FACET_DECOMPOSE` to OFF; the bench adapter
  defaulted it to ON, and the fleet (KAX machines, rogue agents) runs it ON. The adapter
  default flips to OFF with this entry, so the standard row is the shipped default. Whether
  the fleet should keep facets on is a question for a corpus where facets can matter (short
  compound notes, not chat turns) — not answered here.

Honest standard row for `kannaka_minilm`, longmemeval_s, k=15: **1.000 / 0.950 / 0.848 /
0.918, p50 1.0 s, 98 ms/item, 46 KB/item.**

## 2026-09-22 — E-L1b: the fine-tuned reflex passes its rule (held-out AUROC 0.963)

Fine-tuned `convaiinnovations/laya` (English, ModernBERT-large) on Kannaka's evidence-gate
decision with `experiments/laya_reflex/train_e_l1b.py` — upstream's RLCD recipe (zero-mean
noisy-logit groups scored by proper scoring rules + soft cross-entropy) reduced to one GPU.
Training data from `build_e_l1b_dataset.py`: **4,180 (question, turn) rows from the 470
questions outside the standard 30** — 836 positive turns (`has_answer`), 3,344 negatives, of
which 3,320 are hard (same session as a positive). **The 30 standard questions were held out
entirely**; the test set is E-L1's identical 450 decisions over the medium's own top-15
candidates. qBraid RTX 4090, 4 epochs, 390 s, ~5 steps/s; ≈ 40 credits (≈ $0.40) for the
instance including both evaluations.

| model | AUROC | Brier | ECE | precision @0.5 | recall @0.5 | latency / decision (4090) |
|---|---|---|---|---|---|---|
| Laya off the shelf, CPU (E-L1) | 0.750 | 0.092 | 0.057 | 0.25 | 0.19 | 507 ms (CPU) |
| Laya off the shelf, same GPU (control) | 0.748 | 0.092 | 0.057 | 0.25 | 0.19 | 20.6 ms |
| **Laya fine-tuned (E-L1b), held-out** | **0.963** | **0.066** | 0.066 | **0.59** | **0.95** | **20.6 ms** |

The GPU control reproduces E-L1 to the third decimal, so the pipeline measures what it says
it measures; the fine-tune moves AUROC by +0.21 on questions it never saw. Pre-registered rule
was AUROC ≥ 0.85 and Brier ≤ 0.15: **passed**. At p ≥ 0.5 it keeps 41 of the 43 evidence turns
and flags 69 of 450 candidates in total — as a gate it would hand the answer stage ~2.3 rows
per question instead of 15 while keeping 95% of the evidence.

What this does and does not show:
- It shows a **memory substrate can train its own reflex from labels it already has**, in
  seven minutes, for under a dollar, and that the reflex runs at 20 ms on a consumer GPU.
- It does not show the gate improves *answers*: that is the next bench run (answer stage over
  the gated rows vs plain top-15, accuracy and tokens per question, same 30 questions).
- n = 43 positives; the AUROC's standard error is roughly ±0.03. Fine-tuned noul temperature
  came out at 4.98 (heavily softened), which is why ECE stayed flat while Brier improved.
- Negatives were sampled per question from the same haystack, so the model has not seen
  cross-question distractors the way a production store would present them.

The fine-tuned agent directory (`model.safetensors`, `encoder/`, `tokenizer/`,
`rl_agent_config.json`, `train_meta.json`) is in Laya's layout and is what upstream's notebook
pushes to the Hub — publication under kannaka-labs is one call with a token.

## 2026-09-23 — E-L1c: the gate keeps the evidence and still loses answers (a loss)

The fine-tuned reflex keeps 95% of the labelled evidence turns while discarding 85% of the
candidates. Does handing the answer model only those rows improve answers, or at least keep
them at a fraction of the tokens? Pre-registered rule (`experiments/laya_reflex/README.md`):
the gate ships as an answer-stage option if accuracy is within one question of plain top-15
at ≤ half the tokens, or higher at any token count.

**Answer model caveat, stated first.** The Anthropic key behind the lab gateway hit its monthly
limit tonight (Sonnet and Haiku both refuse until 2026-10-01), so both arms ran on
`qwen2.5:14b` — the gateway's `kannaka-brain` route, served from ollama on a qBraid RTX 4090
through an ssh tunnel to keep the answer stage and its data on debain2. Same model for both
arms, same judge (the model judging itself, as the answer stage does by default), same 30
questions, same day. **This is a within-model comparison; the absolute numbers are not the
Sonnet rows above.** (The 14B model scored 0.800 at top-15 under its own judging, above the
0.733 Sonnet-with-Sonnet-judge row; judges differ, so no conclusion about the models is drawn.)

| arm (qwen2.5:14b answers and judges) | accuracy | correct | prompt tok / q | excerpts / q | answer p50 |
|---|---|---|---|---|---|
| plain top-15, turn pairs (the standard setting) | **0.800** | 24 / 30 | 5 163 | 20.7 | 3.2 s |
| gate: candidates with fine-tuned P(yes) ≥ 0.5, min 1, turn pairs | 0.667 | 20 / 30 | **1 453** | 4.7 | 2.4 s |

Six questions flipped, five against the gate, one for it:

```
51a45a95  single-session-user        1 -> 0
6d550036  multi-session              1 -> 0
06878be2  single-session-preference  1 -> 0
6a1eabeb  knowledge-update           1 -> 0
852ce960  knowledge-update           1 -> 0
35a27287  single-session-preference  0 -> 1
```

By type (correct of 5): knowledge-update 5 → 3, multi-session 2 → 1, single-session-user
4 → 3, single-session-preference 4 → 4 (one lost, one gained), temporal 4 → 4,
single-session-assistant 5 → 5.

**Rule: failed** — four questions down at 28% of the tokens. Reading it:
- The gate is measured against LongMemEval's `has_answer` turns, and it finds them. The
  answer model, though, was using turns the labels do not mark: the two knowledge-update
  losses are the clearest case, where the model needs the *superseded* statement as well as
  the current one to say what changed, and only the current one carries `has_answer`. The
  gate is faithful to its labels and the labels are narrower than the answer needs.
- 1 453 tokens per question is where the sibling/pair expansion lands after the gate has
  already cut the candidate list to 2.4 rows; the saving is real (3.6×) and would matter at
  scale, but not at four questions in thirty.
- The one gain (a preference question) is the distraction effect the gate was meant to buy,
  and it bought it once.

What this leaves open, in order of promise: (1) gate with a wider aperture — keep the
top-k *sessions* any kept turn belongs to, rather than the turns, so supersession context
survives; (2) train the reflex on "used by the answer" labels rather than `has_answer`, which
this run produced for the first time (the 24 correct top-15 answers and their excerpts);
(3) re-run both arms on Sonnet when the account limit resets, since a stronger model may
lean on the gate less. None of these is claimed; each is a run.

Cost: the second RTX 4090 pod for the answer model, about 50 minutes including the ollama
install and a 9 GB pull, ≈ 45 qBraid credits (≈ $0.45). Raw rows in
`experiments/laya_reflex/results/e_l1c_*`.
