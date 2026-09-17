# kannaka-bench — the Kannaka Memory Benchmark Suite

Reproducible benchmarks for persistent agent memory. The question is simple and
public: **given a long history, does a memory system bring back the right thing,
how fast, and at what cost?** kannaka is measured next to vector baselines, plain
context, and — where their open-source paths allow — Mem0, Zep/Graphiti and Letta.
**Every loss is published in the same table as every win.**

## What is measured

| axis | metric | how |
|---|---|---|
| retrieval | recall@k, MRR | does an item from the gold evidence session/turn appear in the top-k |
| answer quality | accuracy (LLM-judged), per question type | the answer model sees only the top-k recalled items (phase 2) |
| latency | recall p50 / p95, ingest per item | wall clock, same box, same order, warm |
| footprint | bytes on disk per item, resident memory | store size after ingest |
| cost | ingest and recall tokens / seconds | per 1 000 items |
| scaling | the above at 1×, 4×, 16× history | LongMemEval S vs M, LoCoMo full |

## Datasets

- **LongMemEval** (Wu et al., 2024) — `longmemeval_oracle` (evidence sessions only),
  `longmemeval_s` (~115k tokens of haystack per question), `longmemeval_m` (~1.5M).
  Each question carries its own haystack: one fresh store per question.
- **LoCoMo** (Maharana et al., 2024) — 10 very long two-person conversations, ~2 000
  questions with turn-level evidence (`D1:3`), five question categories.

Loaders download to `~/.kannaka-bench/data/` and record the file sha256 in the run
manifest, so a result names the exact bytes it was measured on.

## Adapters

One interface (`bench/adapters/base.py`): `ingest(items)`, `recall(query, k)`,
`footprint_bytes()`. Items carry an id, text and a timestamp; recall returns ids.

| adapter | what it is | status |
|---|---|---|
| `kannaka` | the kannaka binary as shipped (the default `hash:384` encoder), one HRM store per run, swarm publishing disabled | in |
| `kannaka_minilm` | the same binary with `all-MiniLM-L6-v2` as its encoder (via ollama) — same encoder family as the vector baseline, so the comparison is about the medium | in |
| `vector_numpy` | sentence-transformers `all-MiniLM-L6-v2` + cosine over numpy — the honest vector baseline | in |
| `recency` | no retrieval: the k most recent items (the "plain context" floor for recall@k; full context in phase 2) | in |
| `pgvector` | the same embeddings in Postgres/pgvector | next |
| `mem0`, `zep_graphiti`, `letta` | their open-source local paths | next |

## Running

```
python -m bench.run --dataset longmemeval_oracle --adapters kannaka,vector_numpy,recency --k 5 --limit 50 --out results/
python -m bench.report results/<run>/
```

A run writes `manifest.json` (commit, adapters, dataset sha256, hardware, args) and
`results.jsonl` (one row per question per adapter). `report` prints the table.
Runs are meant for debain2 (20 cores, 196 GB); the kannaka adapter needs the
`kannaka` binary on PATH (`KANNAKA_BIN` to override).

## Rules

1. Same box, same order, same k for every adapter in a run.
2. Nothing is tuned on the test questions. Defaults are the shipped defaults.
3. A number without a manifest is not a result.
4. Losses are reported with the same precision as wins, in the same table.
