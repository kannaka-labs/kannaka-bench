"""Stamp accounting for a supersede map over the held-out knowledge-update questions.

For each question the two `has_answer` turns are the old fact (earlier session) and the new fact
(later session). A map entry on the old id is a TRUE catch; one on the new id EXPIRES A CURRENT
FACT; everything else is a false stamp. Also reports whether the old id was stamped BY the true
new turn (the exact pair) and the 5K flagship case (6a1eabeb).

    python map_stats.py --dataset ku78.json --question-ids ids.txt map_k10.json map_k20.json ...
"""
import argparse
import json
from datetime import datetime


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
    ap.add_argument("--question-ids", required=True)
    ap.add_argument("maps", nargs="+")
    a = ap.parse_args()
    ids = {l.strip() for l in open(a.question_ids) if l.strip()}
    qs = [q for q in json.load(open(a.dataset, encoding="utf-8")) if q["question_id"] in ids]
    pairs = {}
    for q in qs:
        sids, dates, sessions = q["haystack_session_ids"], q.get("haystack_dates") or [], q["haystack_sessions"]
        ev = []
        for i, s in enumerate(sessions):
            for t, turn in enumerate(s):
                if turn.get("has_answer"):
                    ev.append((parse_date(dates[i]) if i < len(dates) else None, f"{sids[i]}#{t}"))
        ev.sort(key=lambda x: (x[0] or datetime.min, x[1]))
        if len(ev) >= 2:
            pairs[q["question_id"]] = (ev[0][1], ev[-1][1])
    print(f"questions {len(qs)}, with an (old,new) evidence pair {len(pairs)}")
    for mp in a.maps:
        m = json.load(open(mp))
        mm = m.get("map", m)
        stats = m.get("stats", {})
        stamped = set(mm)
        true_caught = [qid for qid, (old, new) in pairs.items() if old in stamped]
        exact = [qid for qid, (old, new) in pairs.items() if old in stamped and mm[old].get("by") == new]
        expired_current = [qid for qid, (old, new) in pairs.items() if new in stamped]
        evidence_ids = {x for p in pairs.values() for x in p}
        false = [k for k in stamped if k not in evidence_ids]
        five_k = next((qid for qid in pairs if qid.startswith("6a1eabeb")), None)
        print(f"{mp}: k={m.get('k')} thr={m.get('threshold')} pairs={stats.get('pairs')} stamped={len(stamped)} "
              f"true_caught={len(true_caught)}/{len(pairs)} (exact pair {len(exact)}) "
              f"current_expired={len(expired_current)} false={len(false)} "
              f"5K_caught={five_k in true_caught if five_k else 'n/a'}")
        print("  uncaught:", sorted(q[:8] for q in pairs if q not in true_caught))


if __name__ == "__main__":
    main()
