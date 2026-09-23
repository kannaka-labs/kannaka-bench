"""E-L3: supersession at write time — does a new statement update an earlier one?

LongMemEval-S knowledge-update questions each carry exactly two evidence sessions: the turn
where the user stated a fact and a later turn where the fact changed (70 of 78 are that clean).
That is a labelled supersession pair. The decision, asked of Laya as a `noul`:

    state = {"earlier": "[date] role: text", "later": "[date] role: text"}
    "Does the later statement update, correct or replace a fact stated in the earlier statement?"

Positives: (old evidence turn, new evidence turn). Hard negatives, per question: other turns
from the OLD evidence session paired with the same new turn (same conversation, no update),
topped up with turns from other sessions dated before the new turn. All pairs keep the real
chronological order, so date alone cannot separate them.

Modes:
  --eval        score every pair with the Router (off the shelf) or --agent-dir (fine-tuned);
                writes decisions.jsonl + e_l3_summary.json (AUROC, Brier, ECE, P/R @0.5, latency)
  --build-train write the E-L3b training rows (E-L1b row format) for the questions NOT held out

Held out: the knowledge-update questions of the standard 30 plus the next 15 by sorted id (20).
Decision rules are pre-registered in README.md.
"""
import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np

INSTRUCTIONS = "Does the later statement update, correct or replace a fact stated in the earlier statement?"
Q = {"supersedes": {"type": "noul", "instructions": INSTRUCTIONS}}


def p_yes(ans):
    for k in ("noul", "probability", "p_yes"):
        if k in ans and isinstance(ans[k], (int, float)):
            return float(ans[k])
    raise KeyError(ans)


def auroc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(order))
    sv = vals[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def fmt(date, turn, cap):
    return f"[{date}] {turn.get('role', '?')}: {turn.get('content', '')}"[:cap]


def build_pairs(dataset_path, holdout_results, neg_per_pos, cap, seed):
    """Returns (pairs, held_out_ids, stats). Each pair: dict(question_id, earlier, later, label, earlier_id, later_id)."""
    rng = random.Random(seed)
    data = json.load(open(dataset_path, encoding="utf-8"))
    ku = [q for q in data if q["question_type"] == "knowledge-update"]
    std = {json.loads(l)["question_id"] for l in open(holdout_results, encoding="utf-8") if l.strip()}
    std_ku = sorted(q["question_id"] for q in ku if q["question_id"] in std)
    rest = sorted(q["question_id"] for q in ku if q["question_id"] not in std)
    held = set(std_ku + rest[:20 - len(std_ku)])
    pairs, stats = [], Counter()
    for q in ku:
        sids, dates, sessions = q["haystack_session_ids"], q.get("haystack_dates") or [], q["haystack_sessions"]
        ev = [(i, t) for i, s in enumerate(sessions) for t, turn in enumerate(s) if turn.get("has_answer")]
        ev_sessions = sorted({i for i, _ in ev})
        if len(ev_sessions) != 2 or len(ev) != 2:
            stats["skipped_not_two_evidence_turns"] += 1
            continue
        (si_a, ta), (si_b, tb) = sorted(ev, key=lambda x: (dates[x[0]] if x[0] < len(dates) else "", x[0]))
        old_turn, new_turn = sessions[si_a][ta], sessions[si_b][tb]
        old_date, new_date = dates[si_a] if si_a < len(dates) else "?", dates[si_b] if si_b < len(dates) else "?"
        later = fmt(new_date, new_turn, cap)
        pairs.append({"question_id": q["question_id"], "earlier": fmt(old_date, old_turn, cap), "later": later, "label": 1,
                      "earlier_id": f"{sids[si_a]}#{ta}", "later_id": f"{sids[si_b]}#{tb}"})
        stats["positives"] += 1
        same = [(si_a, t) for t, turn in enumerate(sessions[si_a]) if t != ta and turn.get("content", "").strip()]
        other = [(i, t) for i, s in enumerate(sessions) for t, turn in enumerate(s)
                 if i not in (si_a, si_b) and (dates[i] if i < len(dates) else "") <= new_date and turn.get("content", "").strip()]
        rng.shuffle(same)
        rng.shuffle(other)
        negs = same[:neg_per_pos]
        stats["hard_negatives"] += len(negs)
        if len(negs) < neg_per_pos:
            negs += other[:neg_per_pos - len(negs)]
        for i, t in negs:
            pairs.append({"question_id": q["question_id"], "earlier": fmt(dates[i] if i < len(dates) else "?", sessions[i][t], cap),
                          "later": later, "label": 0, "earlier_id": f"{sids[i]}#{t}", "later_id": f"{sids[si_b]}#{tb}"})
            stats["negatives"] += 1
    return pairs, held, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--holdout-results", required=True, help="results.jsonl of the standard 30-question run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-set", choices=("all", "heldout", "train"), default="all")
    ap.add_argument("--agent-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--build-train", default=None, help="write E-L3b training rows (non-held-out questions) to this path")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs, held, stats = build_pairs(args.dataset, args.holdout_results, args.neg_per_pos, args.cap, args.seed)
    stats["held_out_questions"] = len(held)
    print(json.dumps(stats), flush=True)

    if args.build_train:
        rows = []
        for p in pairs:
            if p["question_id"] in held:
                continue
            rows.append({"question_id": p["question_id"], "earlier_id": p["earlier_id"], "later_id": p["later_id"],
                         "state": {"earlier": p["earlier"], "later": p["later"]},
                         "questions": Q,
                         "gold": {"supersedes": {"probabilities": {"true": float(p["label"]), "false": 1.0 - p["label"]}}}})
        random.Random(args.seed).shuffle(rows)
        with open(args.build_train, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"train rows: {len(rows)} ({sum(1 for r in rows if r['gold']['supersedes']['probabilities']['true'] == 1.0)} positive) -> {args.build_train}", flush=True)

    if not args.eval:
        return
    if args.agent_dir:
        import laya
        agent = laya.Agent(args.agent_dir, device=args.device) if args.device else laya.Agent(args.agent_dir)
        predict = lambda s: agent.predict(s, Q)  # noqa: E731
        tag = "finetuned"
    else:
        from laya import Router
        router = Router(preload=False)
        predict = lambda s: router.predict(s, Q)  # noqa: E731
        tag = "baseline"
    sel = [p for p in pairs if (args.eval_set == "all") or (args.eval_set == "heldout") == (p["question_id"] in held)]
    y, pr, lat = [], [], []
    dec_path = os.path.join(args.out, f"decisions_{tag}_{args.eval_set}.jsonl")
    open(dec_path, "w").close()
    for i, p in enumerate(sel):
        t1 = time.perf_counter()
        res = predict({"earlier": p["earlier"], "later": p["later"]})
        lat.append(time.perf_counter() - t1)
        pp = p_yes(res["answers"]["supersedes"])
        if i == 0:
            print("first raw:", json.dumps(res["answers"]["supersedes"]), flush=True)
        y.append(p["label"])
        pr.append(pp)
        with open(dec_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**{k: p[k] for k in ("question_id", "earlier_id", "later_id", "label")}, "p_yes": round(pp, 4)}) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sel)} median latency {np.median(lat) * 1000:.0f} ms", flush=True)
    y, pr = np.array(y, float), np.array(pr, float)
    pred = pr >= 0.5
    tp = float(((pred == 1) & (y == 1)).sum())
    summ = {"tag": tag, "eval_set": args.eval_set, "n": int(len(y)), "n_positive": int(y.sum()),
            "auroc": auroc(y, pr), "brier": float(((pr - y) ** 2).mean()), "brier_always_no": float((y ** 2).mean()),
            "ece_10": ece(y, pr), "precision_at_0.5": tp / max(1.0, float(pred.sum())), "recall_at_0.5": tp / max(1.0, float(y.sum())),
            "latency_ms_median": float(np.median(lat) * 1000), "held_out_questions": sorted(held)}
    json.dump(summ, open(os.path.join(args.out, f"e_l3_summary_{tag}_{args.eval_set}.json"), "w"), indent=1)
    print("E-L3 summary:", json.dumps({k: v for k, v in summ.items() if k != "held_out_questions"}), flush=True)


if __name__ == "__main__":
    main()
