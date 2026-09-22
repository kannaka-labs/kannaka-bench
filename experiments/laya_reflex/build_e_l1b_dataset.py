"""E-L1b dataset: turn-level evidence labels from LongMemEval-S, in Laya's fine-tune row format.

One row per (question, turn) pair: state = {"question", "excerpt"}, one `noul` question
("does the excerpt contain information needed to answer the question?"), gold = a hard
{"true": 1, "false": 0} or {"true": 0, "false": 1} distribution from the dataset's turn-level
`has_answer` flag. This is exactly the decision E-L1 measured, so the fine-tuned model is
scored by the same script on the same held-out 450 decisions.

Split: every question that appears in the reference run's results (the standard 30) is HELD
OUT entirely — none of its turns, positive or negative, is trained on. The other 470
questions supply all their positive turns and `--neg-per-pos` sampled negatives each, drawn
preferentially from the same sessions as the positives (hard negatives) and topped up from the
rest of that question's haystack. Deterministic under --seed.

    ~/laya-venv/bin/python build_e_l1b_dataset.py \
        --dataset ~/.kannaka-bench/data/longmemeval_s.json \
        --holdout-results ~/kannaka-bench-results/postflip-k15/results.jsonl \
        --out ~/kannaka-bench-results/laya-reflex/e_l1b_train.jsonl --neg-per-pos 4 --seed 1
"""
import argparse
import json
import random
from collections import Counter

INSTRUCTIONS = "Does the excerpt contain information needed to answer the question?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--holdout-results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--max-excerpt-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    holdout = {json.loads(l)["question_id"] for l in open(args.holdout_results, encoding="utf-8") if l.strip()}
    data = json.load(open(args.dataset, encoding="utf-8"))

    rows, stats = [], Counter()
    for q in data:
        qid = q["question_id"]
        if qid in holdout:
            stats["questions_heldout"] += 1
            continue
        stats["questions_train"] += 1
        sids = q.get("haystack_session_ids") or []
        pos, same_session_neg, other_neg = [], [], []
        for i, s in enumerate(q.get("haystack_sessions") or []):
            sid = sids[i] if i < len(sids) else f"s{i}"
            session_has_answer = any(t.get("has_answer") for t in s)
            for t, turn in enumerate(s):
                text = f"{turn.get('role', '?')}: {turn.get('content', '')}"[: args.max_excerpt_chars]
                item = (f"{sid}#{t}", text)
                if turn.get("has_answer"):
                    pos.append(item)
                elif session_has_answer:
                    same_session_neg.append(item)
                else:
                    other_neg.append(item)
        if not pos:
            stats["questions_without_positive_turns"] += 1
            continue
        want = args.neg_per_pos * len(pos)
        rng.shuffle(same_session_neg)
        rng.shuffle(other_neg)
        negs = same_session_neg[:want]
        if len(negs) < want:
            negs += other_neg[: want - len(negs)]
        stats["hard_negatives"] += min(len(same_session_neg), want)
        for label, items in ((True, pos), (False, negs)):
            for tid, text in items:
                rows.append({
                    "question_id": qid,
                    "turn_id": tid,
                    "qtype": q.get("question_type", ""),
                    "state": {"question": q.get("question", ""), "excerpt": text},
                    "questions": {"evidence": {"type": "noul", "instructions": INSTRUCTIONS}},
                    "gold": {"evidence": {"probabilities": {"true": 1.0 if label else 0.0, "false": 0.0 if label else 1.0}}},
                })
                stats["positives" if label else "negatives"] += 1

    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"rows": len(rows), "seed": args.seed, "neg_per_pos": args.neg_per_pos,
                "instructions": INSTRUCTIONS, **stats}
    json.dump(manifest, open(args.out + ".manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
