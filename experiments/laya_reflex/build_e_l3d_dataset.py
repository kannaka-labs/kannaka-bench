"""E-L3d dataset: the supersession decision trained at the WRITE PATH's own distribution.

E-L3b's negatives were other turns from the same conversation as the old fact (1:4). The write
path sees something else: for every new turn, the top-k most similar EARLIER turns by MiniLM
cosine across the whole haystack — cross-topic, cross-session, ~1 true supersession per 2,300
pairs. E-L3c showed a 0.94-precision reflex at 1:4 stamps ~380 false positives at that prior.

So: for each non-held-out knowledge-update question, run exactly the E-L3c shortlist (chronological
walk, k earlier neighbours per turn) and label every pair 0 except the (old evidence turn, new
evidence turn) pair, labelled 1. Negatives are then subsampled to --neg-per-pos per question,
stratified so that half come from the pairs whose cosine is highest (the ones a gate would let
through) and half uniformly. Positives are repeated --pos-repeat times so the optimiser sees them.

    python build_e_l3d_dataset.py --dataset ku78.json --holdout-results postflip-results.jsonl \
        --out e_l3d_train.jsonl --k 5 --neg-per-pos 40 --pos-repeat 6 --device cuda
"""
import argparse
import json
import random
from collections import Counter
from datetime import datetime

import numpy as np

INSTRUCTIONS = "Does the later statement update, correct or replace a fact stated in the earlier statement?"
Q = {"supersedes": {"type": "noul", "instructions": INSTRUCTIONS}}


def parse_date(s):
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--holdout-results", required=True)
    ap.add_argument("--holdout-ids", default=None, help="E-L3g: file of question ids to hold out (one per line); overrides the standard-run hold-out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neg-per-pos", type=int, default=40)
    ap.add_argument("--pos-repeat", type=int, default=6)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.embed_model, device=args.device)

    data = json.load(open(args.dataset, encoding="utf-8"))
    ku = [q for q in data if q["question_type"] == "knowledge-update"]
    std = {json.loads(l)["question_id"] for l in open(args.holdout_results, encoding="utf-8") if l.strip()}
    std_ku = sorted(q["question_id"] for q in ku if q["question_id"] in std)
    rest = sorted(q["question_id"] for q in ku if q["question_id"] not in std)
    held = set(std_ku + rest[:20 - len(std_ku)])   # identical hold-out to E-L3/E-L3c
    if args.holdout_ids:
        held = {l.strip() for l in open(args.holdout_ids) if l.strip()}

    rows, stats = [], Counter()
    for q in ku:
        if q["question_id"] in held:
            continue
        sids, dates, sessions = q["haystack_session_ids"], q.get("haystack_dates") or [], q["haystack_sessions"]
        ev = [(i, t) for i, s in enumerate(sessions) for t, turn in enumerate(s) if turn.get("has_answer")]
        if len(ev) != 2 or len({i for i, _ in ev}) != 2:
            stats["skipped"] += 1
            continue
        items = []
        for i, s in enumerate(sessions):
            d = parse_date(dates[i]) if i < len(dates) else None
            for t, turn in enumerate(s):
                if turn.get("content", "").strip():
                    items.append({"id": f"{sids[i]}#{t}", "when": d, "ds": dates[i] if i < len(dates) else "?",
                                  "text": f"{turn.get('role', '?')}: {turn.get('content', '')}", "ev": bool(turn.get("has_answer"))})
        items.sort(key=lambda it: (it["when"] or datetime.min, it["id"]))
        vecs = emb.encode([it["text"][:1000] for it in items], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        ev_items = [it for it in items if it["ev"]]
        old_id, new_id = ev_items[0]["id"], ev_items[1]["id"]
        pos, negs = None, []
        for j in range(1, len(items)):
            later = items[j]
            earlier_idx = [i for i in range(j) if items[i]["when"] and later["when"] and items[i]["when"] < later["when"]]
            if not earlier_idx:
                continue
            sims = vecs[earlier_idx] @ vecs[j]
            for rank, i in enumerate(np.argsort(-sims)[: args.k]):
                e = items[earlier_idx[i]]
                pair = {"question_id": q["question_id"], "earlier_id": e["id"], "later_id": later["id"], "cos": float(sims[i]),
                        "state": {"earlier": f"[{e['ds']}] {e['text']}"[: args.cap], "later": f"[{later['ds']}] {later['text']}"[: args.cap]}}
                if e["id"] == old_id and later["id"] == new_id:
                    pos = pair
                else:
                    negs.append(pair)
        if pos is None:
            # the true pair was not in the top-k shortlist: add it anyway (it is what the reflex must say yes to)
            e = next(it for it in items if it["id"] == old_id); n = next(it for it in items if it["id"] == new_id)
            pos = {"question_id": q["question_id"], "earlier_id": old_id, "later_id": new_id, "cos": None,
                   "state": {"earlier": f"[{e['ds']}] {e['text']}"[: args.cap], "later": f"[{n['ds']}] {n['text']}"[: args.cap]}}
            stats["positive_not_in_shortlist"] += 1
        stats["shortlist_pairs"] += len(negs) + 1
        negs.sort(key=lambda p: -p["cos"])
        half = args.neg_per_pos // 2
        chosen = negs[:half] + rng.sample(negs[half:], min(len(negs) - half, args.neg_per_pos - half))
        for _ in range(args.pos_repeat):
            rows.append({**pos, "label": 1})
        for p in chosen:
            rows.append({**p, "label": 0})
        stats["questions"] += 1
        stats["positives"] += 1
        stats["negatives"] += len(chosen)
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"question_id": r["question_id"], "earlier_id": r["earlier_id"], "later_id": r["later_id"],
                                "cos": r["cos"], "state": r["state"], "questions": Q,
                                "gold": {"supersedes": {"probabilities": {"true": float(r["label"]), "false": 1.0 - r["label"]}}}},
                               ensure_ascii=False) + "\n")
    stats["rows"] = len(rows)
    stats["held_out"] = len(held)
    json.dump(dict(stats), open(args.out + ".manifest.json", "w"), indent=1)
    print(json.dumps(dict(stats)), flush=True)


if __name__ == "__main__":
    main()
