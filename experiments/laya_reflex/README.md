# Laya as a System-1 reflex for Kannaka — pre-registered experiments

Laya (`NandhaKishorM/laya`, forked to `flaukowski/laya`) answers typed questions — `choice`,
`score`, `noul` — over a state in one non-autoregressive forward pass, calibrated by training
against proper scoring rules. Kannaka today makes its fast decisions with heuristics, a
caller-supplied constant, or an LLM call. These experiments measure whether Laya can take
those decisions, **calibrated first on cases whose answer we already know**, before it is
allowed to touch a store.

Each experiment fixes its decision rule before the run. Wins and losses both get published.

## E-L1 — evidence gate at recall

For every question in a finished bench run, take the top-k candidates the medium actually
retrieved and ask one `noul` per candidate: *does the excerpt contain information needed to
answer the question?* Ground truth: LongMemEval's turn-level `has_answer`.

Reported: AUROC, Brier, ECE (10 bins), precision/recall at p ≥ 0.5, per-decision latency on
the host that ran it.

**Decision rule (fixed 2026-09-22):** Laya becomes a candidate recall gate — worth a full
bench run behind the answer stage — if **AUROC ≥ 0.85 and Brier ≤ 0.15** on the standard 30
questions. Between 0.75 and 0.85 it goes to fine-tuning (Laya ships the typed-decisions
notebook) before any further use. Below 0.75 the gate idea is dropped for this checkpoint and
recorded as a loss.

Latency is reported, not gated: a CPU number above ~150 ms per decision means the gate is
for dreams and membranes, not per-hit recall, until a GPU sits behind it.

## E-L2 — question routing without the gold leak

The bench's answer stage routes prompts on the dataset's `question_type` (a gold-label leak).
Ask Laya a `choice` over the six types from the question text alone.

**Decision rule (fixed 2026-09-22):** accuracy **≥ 0.80** over all 500 questions replaces gold
routing in the answer stage (and the leak is closed). Below that the routing stays off and the
answer stage runs un-routed, which is the honest baseline anyway.

### Outcomes (2026-09-22)

- **E-L1: loss.** AUROC 0.750 (0.748 on GPU), Brier 0.092 against 0.096 for "always no".
- **E-L2: loss.** Accuracy 0.314; the model answers the first criterion for 348 of 500.
- **E-L1b (fine-tune, `build_e_l1b_dataset.py` + `train_e_l1b.py`, RTX 4090, 390 s): PASS.**
  Held-out AUROC **0.963**, Brier 0.066, recall 0.95 / precision 0.59 at p ≥ 0.5, 20.6 ms per
  decision. Same rule as E-L1, same 450 decisions, the 30 questions never trained on.
  Raw outputs: `results/e_l1b_finetuned_heldout.json`, control `results/e_l1_baseline_gpu.json`.

## E-L1c — does the gate improve answers? (next)

Answer stage over the gated candidates (fine-tuned `noul` ≥ 0.5, minimum 1 row) versus the
plain top-15, same 30 questions, same judge: accuracy and tokens per question.
**Decision rule:** the gate ships as an answer-stage option if accuracy is within one question
of top-15 at ≤ half the tokens, or higher at any token count. Otherwise it stays a research row.

**Outcome (2026-09-23): loss.** Within-model on qwen2.5:14b (the Anthropic key behind the
gateway is capped until 2026-10-01): plain top-15 **0.800** (24/30, 5 163 tok/q) vs gated
**0.667** (20/30, 1 453 tok/q). Five flips against, one for; knowledge-update 5 → 3 because the
answer needs the *superseded* turn, which `has_answer` never marks. Rows in
`results/e_l1c_answers_*.jsonl`, gate decisions in `results/e_l1c_gate_decisions.jsonl`.
Next candidates (each a run, none claimed): session-aperture gating; training on
"used-by-the-answer" labels this run produced; the same two arms on Sonnet after the reset.

## E-L3 — supersession at write time (`e_l3_supersession.py`)

LongMemEval-S knowledge-update questions each mark two evidence turns: where a fact was stated
and where it later changed. That is a labelled supersession pair, and the write-time decision
Kannaka never makes today (the temporal triple exists; nothing fills `expires_at`). The
`noul`: *does the later statement update, correct or replace a fact stated in the earlier
statement?* Positives: (old evidence turn, new evidence turn), 70 clean questions. Hard
negatives: 4 per positive, other turns from the same conversation as the old turn paired with
the same new turn, all in true chronological order so date cannot separate them.

**Decision rules (fixed 2026-09-23, before the run):**
- **E-L3a, off the shelf, all 350 pairs:** the control, not the candidate. AUROC ≥ 0.85 and
  Brier ≤ 0.15 would be a surprise worth its own note; otherwise it is the baseline E-L3b is
  measured against. Runs on CPU.
- **E-L3b, fine-tuned on the 50 non-held-out questions (≈250 rows), scored on the 20 held-out
  questions (100 pairs):** the reflex becomes a candidate for the ingest path if held-out
  **AUROC ≥ 0.85 and Brier ≤ 0.15** and recall at p ≥ 0.5 is ≥ 0.80 (a supersession reflex
  that misses a fifth of updates is not one to stamp `expires_at` with). 20 positives is a
  small test; the AUROC's standard error is about ±0.05 and is reported as such.
**Outcomes (2026-09-23):** E-L3a off the shelf AUROC **0.443** over all 340 pairs (0.359 on the
held-out 85) — below chance, Brier = always-no. **E-L3b fine-tuned: held-out AUROC 0.929, Brier
0.039, precision 0.94 / recall 0.88 at p ≥ 0.5, 24.8 ms on an RTX 5090 — PASSED** (17 held-out
positives, so ±0.06). Train-set 0.998 (memorised, as 255 rows in 42 s would be). Model:
`flaukowski/laya-kannaka-supersession`. Files: `results/e_l3_*`, `results/decisions_*`.

**E-L3c outcome (2026-09-23): loss.** Pre-pass stamped 397 of 8,068 turns (found 13/17 true
supersessions, expired 0/17 current facts); answers on qwen2.5:14b: plain **0.765** (13/17) vs
supersede+drop **0.588** (10/17) - two flips for (the 5K time, the textbook case), five against,
one an instrument fault (`--top-k 30` returned 0 rows). Cause: precision collapses at the write
path's 1:2,300 prior. Next: E-L3d (cross-topic negatives, gate p >= 0.99 + same speaker + cosine
floor). Files: `results/e_l3c/`.

**E-L3d outcome (2026-09-23): +1 at zero collateral.** Reflex retrained on the write path's own
shortlist pairs (`build_e_l3d_dataset.py`, 2,346 rows; 19/51 true pairs were outside the top-5
shortlist). Held-out pair AUROC 0.939. Gates A/B/C stamp 42/30/15 (29/18/8 false), all three
answer **0.824** (14/17) vs plain 0.765 vs E-L3c 0.588; the one gain is a different question and
the 5K flagship case is still missed. Carry gate A. Next: k=10/20 shortlist, union negatives.
kannaka-memory #1044 landed (PR #1046, unreleased): recall itself now drops a memory whose
`expires_at` is at or before the as-of instant, so `BENCH_DROP_EXPIRED` becomes a `--at` pass once a
release carries it. Files: `results/e_l3d/`.

- **E-L3c (as run):** ingest arm — at write time, recall the top-5 from the
  store so far, ask the reflex per candidate, stamp `expires_at` on a candidate at p ≥ 0.5;
  measure knowledge-update retrieval and answers against the standard row.

## Planned, not started

- ~~E-L3 supersession at write time~~ → above. (`noul` "does this update an existing memory?", then
  `choice` over the recall shortlist for which one) — measured on knowledge-update questions.
- **E-L4 retrieval-grounded calibration of forgetting** — a dream asks `choice {keep, ghost,
  merge}` per row; the reactivation sidecar is the ground truth that arrives later. Needs the
  #977 sidecar path (shipped in kannaka-memory 0.16.11) and weeks of horizon.
- **The stitching** — a learned router with skip paths between organs (substrate recall, Laya
  reflexes, kannaka-brain, Jev), trained on GPU from the decision traces the above produce.
  Not designable until E-L1/E-L2 say what the reflexes are worth.

## Run

    ~/laya-venv/bin/python experiments/laya_reflex/e_l1_evidence_gate.py \
        --dataset ~/.kannaka-bench/data/longmemeval_s.json \
        --results ~/kannaka-bench-results/postflip-k15/results.jsonl \
        --adapter kannaka_minilm --out ~/kannaka-bench-results/laya-reflex
