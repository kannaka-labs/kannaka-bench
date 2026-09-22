"""E-L1 / E-L2: Laya as a System-1 reflex for Kannaka, calibrated on a case whose answer we know.

E-L1 (evidence gate): for every question in a finished kannaka-bench run, take the top-k
candidates the medium actually retrieved, and ask Laya one `noul` question per candidate:
"does this excerpt carry information needed to answer the question?". Ground truth is
LongMemEval's turn-level `has_answer` flag. Reports AUROC, Brier, ECE, precision/recall at
p=0.5, and per-decision latency on THIS host (the README's 33 ms is a T4 number).

E-L2 (routing without the gold leak): for every question in the dataset, ask Laya a `choice`
over the six LongMemEval question types from the question text alone. Ground truth is the
dataset's `question_type`. The bench answer stage currently routes on the gold type.

Decision rules are pre-registered in README.md next to this file. Run:

    ~/laya-venv/bin/python e_l1_evidence_gate.py \
        --dataset ~/.kannaka-bench/data/longmemeval_s.json \
        --results ~/kannaka-bench-results/postflip-k15/results.jsonl \
        --adapter kannaka_minilm --out ~/kannaka-bench-results/laya-reflex
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

QTYPES = {
    "single-session-user": "the answer is a fact the USER stated in one session",
    "single-session-assistant": "the answer is something the ASSISTANT said or produced in one session",
    "single-session-preference": "the answer depends on a preference the user expressed in one session",
    "multi-session": "the answer needs facts spread across several sessions",
    "temporal-reasoning": "the answer needs dates, durations or the order of events",
    "knowledge-update": "a fact changed later; the answer is the most recent version",
}

EVIDENCE_Q = {
    "evidence": {
        "type": "noul",
        "instructions": "Does the excerpt contain information needed to answer the question?",
    }
}

ROUTE_Q = {
    "qtype": {
        "type": "choice",
        "instructions": "Which kind of memory question is this?",
        "criteria": QTYPES,
    }
}


def p_yes(ans: dict) -> float:
    """Laya's noul answer shape, read defensively (probability of YES).

    Observed on laya 0.3.6: {'type': 'noul', 'noul': 0.0941, 'confidence': 0.9059, ...}
    where `noul` is P(yes) and `confidence` is the calibrated max-class confidence.
    """
    for k in ("noul", "probability", "p_yes", "yes"):
        if k in ans and isinstance(ans[k], (int, float)):
            return float(ans[k])
    probs = ans.get("probabilities") or {}
    for k in ("yes", "true", "True", "1"):
        if k in probs:
            return float(probs[k])
    if "answer" in ans and isinstance(ans["answer"], bool):
        c = float(ans.get("confidence", 0.5))
        return c if ans["answer"] else 1.0 - c
    raise KeyError("cannot read a yes-probability from %r" % (ans,))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-sum (Mann-Whitney) AUROC with ties at 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        total += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(total)


def load_dataset(path: str):
    data = json.load(open(path, encoding="utf-8"))
    qs = {}
    for q in data:
        sids = q.get("haystack_session_ids") or []
        turns = {}
        for i, s in enumerate(q.get("haystack_sessions") or []):
            sid = sids[i] if i < len(sids) else f"s{i}"
            for t, turn in enumerate(s):
                turns[f"{sid}#{t}"] = {
                    "text": f"{turn.get('role', '?')}: {turn.get('content', '')}",
                    "has_answer": bool(turn.get("has_answer")),
                }
        qs[q["question_id"]] = {
            "question": q.get("question", ""),
            "qtype": q.get("question_type", ""),
            "turns": turns,
        }
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", required=True, help="results.jsonl of a finished bench run")
    ap.add_argument("--adapter", default="kannaka_minilm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="laya checkpoint override (english|multilingual|typed-decisions)")
    ap.add_argument("--agent-dir", default=None,
                    help="E-L1b: score a fine-tuned agent directory (laya.Agent) instead of the Router")
    ap.add_argument("--device", default=None, help="cuda|cpu for --agent-dir (default: laya's choice)")
    ap.add_argument("--route-limit", type=int, default=0, help="E-L2: cap questions (0 = all)")
    ap.add_argument("--skip-route", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    if args.agent_dir:
        import laya  # noqa: E402

        agent = laya.Agent(args.agent_dir, device=args.device) if args.device else laya.Agent(args.agent_dir)

        class _Fixed:
            """Router-shaped wrapper so the rest of the script is unchanged."""

            def predict(self, state, questions, model=None):
                return agent.predict(state, questions)

        router = _Fixed()
    else:
        from laya import Router  # noqa: E402

        router = Router(preload=False)
    qs = load_dataset(args.dataset)
    print(f"dataset: {len(qs)} questions; router ready in {time.time() - t0:.1f}s", flush=True)

    # ---------------- E-L1: evidence gate over the medium's own candidates ----------------
    rows = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("adapter") == args.adapter and r.get("candidates")]
    y, p, lat = [], [], []
    per_q = []
    first_raw = None
    for r in rows:
        q = qs.get(r["question_id"])
        if not q:
            continue
        n_pos = n_pred = 0
        for cid in r["candidates"]:
            turn = q["turns"].get(cid)
            if turn is None:
                continue
            state = {"question": q["question"], "excerpt": turn["text"][:2000]}
            t1 = time.perf_counter()
            res = router.predict(state, EVIDENCE_Q, model=args.model)
            lat.append(time.perf_counter() - t1)
            if first_raw is None:
                first_raw = res["answers"]["evidence"]
                print("first raw noul answer:", json.dumps(first_raw), flush=True)
            pp = p_yes(res["answers"]["evidence"])
            y.append(1 if turn["has_answer"] else 0)
            p.append(pp)
            n_pos += turn["has_answer"]
            n_pred += pp >= 0.5
        per_q.append({"question_id": r["question_id"], "qtype": r.get("qtype"), "n": len(r["candidates"]),
                      "gold_turns_in_candidates": n_pos, "predicted_yes": n_pred})
        print(f"E-L1 {r['question_id']} {r.get('qtype')}: {n_pos} gold turns among {len(r['candidates'])}, "
              f"laya said yes to {n_pred}; median latency so far {np.median(lat) * 1000:.0f} ms", flush=True)

    y_arr, p_arr = np.array(y, dtype=float), np.array(p, dtype=float)
    pred = p_arr >= 0.5
    tp = float(((pred == 1) & (y_arr == 1)).sum())
    e_l1 = {
        "n_decisions": int(len(y_arr)),
        "n_positive": int(y_arr.sum()),
        "positive_rate": float(y_arr.mean()) if len(y_arr) else None,
        "auroc": auroc(y_arr, p_arr) if len(y_arr) else None,
        "brier": float(((p_arr - y_arr) ** 2).mean()) if len(y_arr) else None,
        "ece_10": ece(y_arr, p_arr) if len(y_arr) else None,
        "precision_at_0.5": tp / max(1.0, float(pred.sum())),
        "recall_at_0.5": tp / max(1.0, float(y_arr.sum())),
        "latency_ms": {
            "median": float(np.median(lat) * 1000) if lat else None,
            "p95": float(np.percentile(lat, 95) * 1000) if lat else None,
            "mean": float(np.mean(lat) * 1000) if lat else None,
        },
        "per_question": per_q,
    }
    json.dump(e_l1, open(os.path.join(args.out, "e_l1_evidence_gate.json"), "w"), indent=1)
    print("E-L1 summary:", json.dumps({k: v for k, v in e_l1.items() if k != "per_question"}), flush=True)

    # ---------------- E-L2: question-type routing from the question text alone ----------------
    if args.skip_route:
        return
    items = list(qs.items())
    if args.route_limit:
        items = items[: args.route_limit]
    conf = Counter()
    correct = 0
    lat2 = []
    by_type = defaultdict(lambda: [0, 0])
    for qid, q in items:
        t1 = time.perf_counter()
        res = router.predict({"question": q["question"]}, ROUTE_Q, model=args.model)
        lat2.append(time.perf_counter() - t1)
        got = res["answers"]["qtype"]["choice"]
        conf[(q["qtype"], got)] += 1
        by_type[q["qtype"]][1] += 1
        if got == q["qtype"]:
            correct += 1
            by_type[q["qtype"]][0] += 1
    e_l2 = {
        "n": len(items),
        "accuracy": correct / max(1, len(items)),
        "chance": 1.0 / len(QTYPES),
        "by_type": {k: {"correct": v[0], "n": v[1], "acc": v[0] / max(1, v[1])} for k, v in by_type.items()},
        "confusion": [{"gold": g, "pred": pr, "n": n} for (g, pr), n in sorted(conf.items())],
        "latency_ms": {"median": float(np.median(lat2) * 1000) if lat2 else None},
    }
    json.dump(e_l2, open(os.path.join(args.out, "e_l2_qtype_routing.json"), "w"), indent=1)
    print("E-L2 summary:", json.dumps({k: v for k, v in e_l2.items() if k != "confusion"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
