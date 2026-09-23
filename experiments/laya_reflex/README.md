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
supersede+drop **0.647** (11/17; corrected 09-23 — the "instrument fault" row was this harness's
parser splitting a JSON line at U+2028, kannaka-bench f33a542) - two flips for (the 5K time, the
textbook case), four against. Cause: precision collapses at the write
path's 1:2,300 prior. Next: E-L3d (cross-topic negatives, gate p >= 0.99 + same speaker + cosine
floor). Files: `results/e_l3c/`.

**E-L3d outcome (2026-09-23): +1 at zero collateral.** Reflex retrained on the write path's own
shortlist pairs (`build_e_l3d_dataset.py`, 2,346 rows; 19/51 true pairs were outside the top-5
shortlist). Held-out pair AUROC 0.939. Gates A/B/C stamp 42/30/15 (29/18/8 false), all three
answer **0.824** (14/17) vs plain 0.765 vs E-L3c 0.647 (corrected); the one gain is a different question and
the 5K flagship case is still missed. Carry gate A. Next: k=10/20 shortlist, union negatives.
kannaka-memory #1044 landed (PR #1046, unreleased): recall itself now drops a memory whose
`expires_at` is at or before the as-of instant, so `BENCH_DROP_EXPIRED` becomes a `--at` pass once a
release carries it. Files: `results/e_l3d/`.

**E-L3e (pre-registered 2026-09-23, before the run): write-path shortlist k=10 and k=20.** Same 17 held-out questions, same reflex (E-L3d, gate A: p >= 0.5, no other gates), only the shortlist widens from the top-5 earlier neighbours to the top-10 and top-20 (`e_l3c_supersede.py --k`). Motivation: 19/51 training-set true pairs sat outside the top 5, and the 5K flagship case is among the 4 held-out misses. Predictions: true catches rise above 13/17 at k=20 and the 5K case is caught; false stamps grow about linearly with pairs (29 at k=5 -> ~60 at k=10 -> ~120 at k=20) because the per-pair false rate is ~0.07%; current facts expired stays 0. Decision rule: carry the largest k whose answers are >= 0.824 (14/17) with 0 current facts expired; a k whose answers fall below plain 0.765 is a loss and is published as one. Cost: one RTX 6000 Ada pod (~$0.92/h), answers on qwen2.5:14b as before. Noise floor: +-1 question.

**E-L3e outcome (2026-09-23): null on answers.** k=10: 48 stamps, 14/17 true catches, 0 current facts expired, 35 false, answers **0.824** (14/17), question-identical to E-L3d A. k=20: 61 stamps, 14/17, **1 current fact expired** (an unrelated turn at p 0.91 / cos 0.30), 47 false, answers 0.765 (13/17, lost the shoe-rack question to the old location). False stamps grew sub-linearly (30/35/47, not the predicted 60/120). The 5K case is uncaught at every k although its pair is rank 1 by cosine: a classifier miss, not a shortlist miss. Rule says carry k=10; it buys nothing. Next: union-negatives retrain at k=10. Files: `results/e_l3e/`, `map_stats.py`. Cost $1.71 (RTX 6000 Ada).

**E-L3f (pre-registered 2026-09-23, before the run): union negatives.** Retrain the reflex from the base Laya checkpoint on the E-L3d rows (51 positives x6, 2,040 write-path shortlist negatives) plus the E-L3b same-conversation negatives (204 rows at 1:4), same recipe (6 epochs, grad-accum 4), then the k=10 pre-pass (gate A) on the same 17 held-out questions and the same answer model. Motivation: E-L3e showed the 5K pair (6a1eabeb) is rank 1 by cosine in every shortlist and the E-L3d reflex says no to it while the E-L3b reflex said yes; the same-conversation negatives are what taught E-L3b to separate that pair. Predictions: held-out pair AUROC >= 0.93; the 5K pair gets p >= 0.5 (the named success criterion); true catches >= 14/17; false stamps <= 45; current facts expired 0; answers >= 0.824 with the 5K question flipping to correct (15/17 if nothing else moves). Decision rule: carry the union model if answers >= 0.824 and 0 current facts expired; below 0.824 is a loss and is published as one; the 5K catch is reported either way. Cost: one RTX 6000 Ada pod, about $2. Noise floor: +-1 question.

**E-L3f outcome (2026-09-23): loss.** Union model: held-out pair AUROC **0.853** (E-L3b 0.929, E-L3d 0.939), P 1.00 / R 0.65 at 0.5; 5K pair P(yes) 0.309 (from 0.000; still under the stamp threshold). k=10 pre-pass: 24 stamps, 11/17 true catches, 0 current facts expired, 13 false; answers **0.765** (13/17) vs E-L3d 0.824. Four of six predictions missed. The two negative distributions made the model conservative rather than sharper. Line verdict: E-L3d gate A (+1, inside noise) is the best reflex; the flagship case is caught only by the imprecise model. Next: widen the held-out set before any more training; #1044 for the recall-side drop. Cost $2.25 (L40S). Files: `results/e_l3f/`.

**E-L3g (pre-registered 2026-09-23, before the run; Nick: "top up qbraid and run k-fold"): 5-fold cross-validation over all 78 LongMemEval knowledge-update questions.** Folds are round-robin over the sorted question ids (16/16/16/15/15, deterministic, `fold_*.txt`). Per fold: the E-L3d recipe unchanged (write-path shortlist negatives at k=5, 40 per positive, positives x6, 6 epochs, grad-accum 4, base Laya checkpoint), trained on the other four folds (`build_e_l3d_dataset.py --holdout-ids`), then the gate-A k=5 pre-pass on the held-out fold; the supersede+drop retrieval arm and the plain arm on the same fold; answers on qwen2.5:14b. Pooled over 78 questions the noise floor is about +-1.3 points instead of +-6. Predictions: pooled supersede+drop answers within +-3 questions of plain (the 17-question +1 does not scale); current facts expired <= 2 in total; false stamps ~30 per 17 questions. Decision rule: a gain is claimed only if the pooled delta is >= +4 questions AND current facts expired <= 2; a pooled delta <= -4 is a loss; anything between is null, and the line is parked either way in favour of the recall-side rule (kannaka-memory #1044). Cost: one RTX 6000 Ada or L40S pod, ~2.5 h GPU (~$2.30-5.70) + debain2 CPU for the retrieval arms; needs the qBraid balance topped up first (267.9 credits at pre-registration).

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
