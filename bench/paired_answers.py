"""Paired, per-question comparison of two adapters' ANSWERS in one run.

A tie in answer accuracy says nothing about whether the two systems got the same questions right.
This prints the 2x2 table (both / only A / only B / neither), an exact McNemar test on the
discordant pairs, and, for each discordant question, how much of the gold evidence each adapter
retrieved — which separates "the extra evidence turned into an answer" from "same evidence, the
answer step differed".

    python -m bench.paired_answers <run_dir> [--answers answers-v2.jsonl] [--a kannaka_minilm] [--b vector_numpy]

Added 2026-09-25 after two readers of the public results (calder and unspent on 1F916) asked the
question this answers: are they the same 22?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (binomial, p=0.5)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def table(correct_a: dict[str, bool], correct_b: dict[str, bool]) -> dict:
    qs = sorted(set(correct_a) & set(correct_b))
    out = {"n": len(qs), "both": [], "only_a": [], "only_b": [], "neither": []}
    for q in qs:
        a, b = correct_a[q], correct_b[q]
        key = "both" if a and b else "only_a" if a else "only_b" if b else "neither"
        out[key].append(q)
    out["p"] = mcnemar_exact(len(out["only_a"]), len(out["only_b"]))
    return out


def _evidence(run: str, adapters: tuple[str, str]) -> dict[tuple[str, str], tuple[int, int]]:
    """(adapter, question) -> (gold evidence turns retrieved at k, gold evidence turns)."""
    from bench.evidence_coverage import questions_for  # needs the dataset cache; optional
    man = json.load(open(os.path.join(run, "manifest.json"), encoding="utf-8"))
    qmap = questions_for(man)
    out = {}
    for line in open(os.path.join(run, "results.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("adapter") not in adapters or "error" in r or not r.get("gold"):
            continue
        q = qmap.get(r["question_id"])
        if not q:
            continue
        ev = {it.id for it in q.items if (it.meta or {}).get("has_answer")
              and (it.id.rpartition("#")[0] if "#" in it.id else it.id) in q.gold_ids}
        if ev:
            out[(r["adapter"], r["question_id"])] = (len(set(r["hits"][:r["k"]]) & ev), len(ev))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run")
    p.add_argument("--answers", default="answers-v2.jsonl")
    p.add_argument("--a", default="kannaka_minilm")
    p.add_argument("--b", default="vector_numpy")
    p.add_argument("--no-evidence", action="store_true", help="skip the evidence columns (no dataset cache needed)")
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in open(os.path.join(a.run, a.answers), encoding="utf-8") if l.strip()]
    ca = {r["question_id"]: bool(r["correct"]) for r in rows if r["adapter"] == a.a}
    cb = {r["question_id"]: bool(r["correct"]) for r in rows if r["adapter"] == a.b}
    qtype = {r["question_id"]: r.get("qtype", "") for r in rows}
    t = table(ca, cb)
    print(f"{a.a} vs {a.b} on {t['n']} questions: both {len(t['both'])} | only {a.a} {len(t['only_a'])} | "
          f"only {a.b} {len(t['only_b'])} | neither {len(t['neither'])} | McNemar exact p = {t['p']:.3f}")
    ev = {} if a.no_evidence else _evidence(a.run, (a.a, a.b))
    for label, qs in (("only " + a.a, t["only_a"]), ("only " + a.b, t["only_b"])):
        for q in qs:
            ea, eb = ev.get((a.a, q)), ev.get((a.b, q))
            evs = f" | evidence {a.a} {ea[0]}/{ea[1]}, {a.b} {eb[0]}/{eb[1]}" if ea and eb else ""
            print(f"  {label}: {q} ({qtype.get(q)}){evs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
