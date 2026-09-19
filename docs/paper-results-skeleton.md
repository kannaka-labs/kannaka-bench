# Results skeleton — *Holographic Resonance Memory: An Interference-Based Memory Architecture for Persistent AI Agents*

Working notes for the paper's evaluation section, written against `RESULTS.md` as of 2026-09-18.
Every number below has a run directory on debain2 and a section in RESULTS.md. Nothing here is a
claim the tables do not support; the losses are in the same list as the wins.

## 1. What is measured

- Three datasets: LongMemEval (`oracle`, `s`; `m` blocked, see §5), LoCoMo (loader only so far),
  ConvoMem (six categories, pre-mixed haystacks). Stratified question samples per type.
- Two stages, always reported together: **retrieval** (session hit@k, recall@k, MRR, and
  turn-level *evidence coverage*: the fraction of answer-bearing turns in the top-k) and **answer
  accuracy** (an answer model over the top-k with turn pairs in time order; LongMemEval-style
  judge; every call's tokens recorded).
- Cost columns beside every quality column: recall p50/p95 ms, ingest ms/item, bytes/item,
  prompt tokens per question, dollars per run.
- Rows: the medium (kannaka) with each encoder; exact cosine over the *same* embeddings (the
  control that isolates the medium from the encoder); a recency baseline; plain context; and
  competitors on the same questions (Supermemory documents mode; memories mode pending, §5).

## 2. Findings that hold at n=60 (LongMemEval-S, k=15)

| claim | evidence | strength |
|---|---|---|
| The shipped default encoder (hash) was the loss: 0.53 hit@k vs 0.93 with MiniLM on the same medium. | `s-5pertype-all3` vs `s-5pertype-kannaka-minilm2` | strong (0.4 gap, mechanism traced: `KANNAKA_ENCODER` default, kannaka-memory #976) |
| The medium retrieves at parity with exact cosine on identical embeddings. | MiniLM: 0.933 vs 0.967 hit, 0.828 vs 0.861 recall (n=30); bge: 0.983 vs 1.000, 0.960 vs 0.969 (n=60) | strong (never more than one question in thirty apart) |
| Final accuracy is at parity: 0.80 vs 0.78 (bge, n=60); 0.733 vs 0.733 (MiniLM, n=30). | `s-10pertype-k15-bge`, `s-5pertype-k15` | strong for parity; the +0.017 is noise |
| Turn-level evidence coverage is a whisker ahead: 0.859 vs 0.852 (bge, n=60); 0.848 vs 0.809 (MiniLM, n=30). | `bench.evidence_coverage` | weak (sits inside noise; the MiniLM gap shrank on bge) |
| The answer stage moved accuracy more than any index change: 0.47 → 0.63 (pairs, order, latest-wins) → 0.73 (k 5→15) → 0.80 (bge). | phase-2 v1/v2, k=15, bge tables | strong |
| Plain context is worse than top-15 retrieval at this history length: 0.27 at 29k tokens vs 0.73 at 5k. | phase-2 v2 | strong |
| Widening the window with session diversity (cap 1) raises recall@15 to 0.983 and cuts accuracy to 0.50: diversity trades against within-session coverage. | cap sweep + shape ablation | strong (a published loss) |
| Neither facets nor the ξ-diversity reranker change retrieval on these sets. | probes `probe-xi`, `probe-facet` | moderate (6-question probes, hit-for-hit identical) |

## 3. Competitor row

| claim | evidence |
|---|---|
| Supermemory (self-hosted, documents mode, same MiniLM weights): hit@15 0.897, recall 0.822, evidence coverage 0.657, accuracy 0.621 — behind both turn-level systems on every quality column; faster search (134 ms) and smaller prompts (3.2k tokens). | `s-5pertype-k15-supermemory2`, n=29 + 1 error row |
| Its default encoder (bge-base) does not explain the gap: on the same weights the turn-level systems are at ≥ 0.967 / 0.942. | bge rows |
| Operational: an LLM call per document even in chunk-only mode; a 10,000-document licence cap; an ingest pause at the default memory limit; a 10× slowdown over a run with internal KV errors. | RESULTS.md notes; reproducible |
| Its published "95% Recall@15" is not reproduced on the retrieval-only path; the memories mode (LLM extraction) is unmeasured pending a costed run. | — |

## 4. Losses, stated as such

- **Cost.** Recall p50 2.7–3.1 s vs 14–32 ms (100×); ingest 370–790 ms/item vs 3–99 (the
  per-recall save path, kannaka-memory #977; the process-per-call shape).
- **Scaling.** Ingest cost per item grows linearly with store size (≈ −74 + 0.77·n ms/item ⇒ O(n²)
  per store; recall ≈ 11.7 ms per item). A 5,192-turn store did not finish in 2 h 8 min. This is why
  LongMemEval-M and ConvoMem beyond context ~20 are not in the tables (#978).
- **Multi-evidence aggregation.** LongMemEval multi-session: all-evidence@15 = 0.50 for *both*
  systems at n=10, accuracy 5 vs 4 of 10. ConvoMem c20: all-evidence@15 = 13–15 % (four evidence
  turns per question), accuracy 0–2 of 5 on user / changing / assistant-facts for both. Widening k
  does not fix it.
- **No retrieval advantage from the medium itself** on a strong encoder. The interference medium
  is a faithful vector store here (the codebook projection is lossless, the recall path ≈ cosine).

## 5. The claim the paper has to make or drop — **first measurement is negative (2026-09-19)**

The consolidation arm now exists and has run (RESULTS.md, `s-5pertype-k15-bge-dream`). One deep
dream cycle between ingest and recall left hit@15, recall@15, evidence coverage, all-evidence and
final accuracy **identical** to the same run without it, cost 141 s per ~500-turn store and +32 %
bytes, and put a synthesized memory at rank 1 in 24 of 30 questions (MRR 0.950 → 0.529). That is
the paper's distinguishing mechanism, measured, not paying for itself on this benchmark. It also
reproduces the dream-attractor pathology of kannaka-memory #963 on a clean corpus in a single
cycle. One open thread in its favour: recall was 2.3× faster afterwards.

The abstract this supports today: *a persistent-agent memory at parity with exact cosine on
retrieval quality and answer accuracy, whose consolidation mechanism is measured and currently
costs more than it returns.* What could still change it: repeated cycles over a long-lived store
(every store here is one-shot, ingested and queried once), excluding dream rows from recall, or a
consolidation that emits joins over evidence rather than cross-cluster syntheses.

### Original framing (kept for the record)

The architecture's stated advantage — a memory that consolidates ("dreams"), forgets on purpose,
and joins evidence *before* the question — is exactly what would move the two open losses
(multi-evidence aggregation; cost through a smaller live store). None of the tables above exercise
consolidation: every store is fresh, ingested once, queried once. The experiment that decides the
paper's central claim is a consolidation arm: ingest, run the dream/consolidation cycle, then
recall — reported against the same cosine control, on the multi-session and ConvoMem multi-evidence
questions, with the store's size and the recall latency after consolidation in the cost columns.
Until that arm exists, the honest abstract is: *parity with exact cosine on quality, a working
persistent-agent memory with a real encoder, at a latency and scaling cost that is measured and
open.*

Blocked / pending: LongMemEval-M (needs #978), LoCoMo run, Supermemory memories mode (spend
decision), Mem0 / Zep / Letta / pgvector adapters, 10-run averages for the numbers within noise.
