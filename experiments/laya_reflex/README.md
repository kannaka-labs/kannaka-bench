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

## Planned, not started

- **E-L3 supersession at write time** (`noul` "does this update an existing memory?", then
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
