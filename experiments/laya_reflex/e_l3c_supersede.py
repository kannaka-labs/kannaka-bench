"""E-L3c pre-pass: which memories does each new memory supersede?

For every question, walk its haystack turns in chronological order. For each turn (the
"later" statement) take the top-k most similar EARLIER turns by MiniLM cosine (the shortlist
a write-time recall would produce) and ask the fine-tuned supersession reflex per pair. A
pair at P(yes) >= --threshold marks the earlier turn as superseded: it gets `expires` = the
later turn's timestamp. Output is one JSON map, item id -> {"expires": iso, "by": later id,
"p": P(yes)}, keyed globally by item id (session#turn), which the bench's kannaka adapter
reads through BENCH_SUPERSEDE_MAP at ingest (stamps `expires`) and at recall
(BENCH_DROP_EXPIRED=1 drops hits expired before the question was asked).

    python e_l3c_supersede.py --dataset ku.json --question-ids ids.txt --agent-dir laya_e_l3b \
        --device cuda --k 5 --threshold 0.5 --out supersede_map.json
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np

INSTRUCTIONS = "Does the later statement update, correct or replace a fact stated in the earlier statement?"
Q = {"supersedes": {"type": "noul", "instructions": INSTRUCTIONS}}


def parse_date(s):
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--question-ids", required=True, help="file with one question_id per line")
    ap.add_argument("--agent-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    # E-L3d gates (all off by default so E-L3c stays reproducible): a probability floor,
    # same speaker on both sides (a user's fact is superseded by the user, not the assistant),
    # and a cosine floor on the shortlist pair.
    ap.add_argument("--same-speaker", action="store_true")
    ap.add_argument("--min-cos", type=float, default=0.0)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    import laya
    from sentence_transformers import SentenceTransformer

    ids = {l.strip() for l in open(args.question_ids) if l.strip()}
    data = [q for q in json.load(open(args.dataset, encoding="utf-8")) if q["question_id"] in ids]
    agent = laya.Agent(args.agent_dir, device=args.device)
    emb = SentenceTransformer(args.embed_model, device=args.device)
    supersede, stats = {}, {"questions": len(data), "pairs": 0, "stamped": 0, "items": 0}
    t0 = time.time()
    for qi, q in enumerate(data):
        sids, dates, sessions = q["haystack_session_ids"], q.get("haystack_dates") or [], q["haystack_sessions"]
        items = []
        for i, s in enumerate(sessions):
            d = parse_date(dates[i]) if i < len(dates) else None
            for t, turn in enumerate(s):
                if not turn.get("content", "").strip():
                    continue
                items.append({"id": f"{sids[i]}#{t}", "when": d, "date_str": dates[i] if i < len(dates) else "?",
                              "text": f"{turn.get('role', '?')}: {turn.get('content', '')}"})
        items.sort(key=lambda it: (it["when"] or datetime.min, it["id"]))
        stats["items"] += len(items)
        vecs = emb.encode([it["text"][:1000] for it in items], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        for j in range(1, len(items)):
            later = items[j]
            earlier_idx = [i for i in range(j) if items[i]["when"] is not None and later["when"] is not None
                           and items[i]["when"] < later["when"]]
            if not earlier_idx:
                continue
            sims = vecs[earlier_idx] @ vecs[j]
            order = np.argsort(-sims)[: args.k]
            top = [(earlier_idx[i], float(sims[i])) for i in order]
            for i, cos in top:
                earlier = items[i]
                if cos < args.min_cos:
                    stats["skipped_cos"] = stats.get("skipped_cos", 0) + 1
                    continue
                if args.same_speaker and earlier["text"].split(":", 1)[0] != later["text"].split(":", 1)[0]:
                    stats["skipped_speaker"] = stats.get("skipped_speaker", 0) + 1
                    continue
                res = agent.predict({"earlier": f"[{earlier['date_str']}] {earlier['text']}"[: args.cap],
                                     "later": f"[{later['date_str']}] {later['text']}"[: args.cap]}, Q)
                p = float(res["answers"]["supersedes"]["noul"])
                stats["pairs"] += 1
                if p >= args.threshold:
                    prev = supersede.get(earlier["id"])
                    # earliest superseding statement wins
                    if prev is None or later["when"] < parse_date(prev["expires_src"]):
                        supersede[earlier["id"]] = {"expires": later["when"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                    "expires_src": later["date_str"], "by": later["id"], "p": round(p, 4),
                                                    "cos": round(cos, 4), "question_id": q["question_id"]}
                        stats["stamped"] += 1
        print(f"[{qi + 1}/{len(data)}] {q['question_id']} items={len(items)} pairs so far={stats['pairs']} "
              f"stamped={len(supersede)} {time.time() - t0:.0f}s", flush=True)
    json.dump({"threshold": args.threshold, "k": args.k, "agent_dir": args.agent_dir, "stats": stats, "map": supersede},
              open(args.out, "w"), indent=1)
    print("done:", json.dumps(stats), "unique stamped items:", len(supersede), flush=True)


if __name__ == "__main__":
    main()
