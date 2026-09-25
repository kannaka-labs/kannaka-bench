"""Confidence intervals for a run: per adapter, per question type, and paired
differences between two adapters on the same questions.

  python -m bench.stats results/<run>/ [--pair kannaka_minilm,vector_numpy] [--ids FILE] [--json out.json]

Why this exists: every table before 2026-09-25 was n=30 (five per type), where
one question moves a type's hit rate by 0.20. A mean without its interval
cannot say which differences are real.

- hit@k (a proportion): Wilson 95% interval.
- recall@k, evid@k, MRR (means of per-question scores in [0, 1]): percentile
  bootstrap over questions, 95%, seeded.
- paired A-B: bootstrap of the per-question difference over the questions both
  adapters answered, plus the win / loss / tie counts. A paired interval that
  excludes 0 is the only thing reported as a difference.

Only scored rows count (no error, non-empty gold), as in `report.py`; evid@k
only where the question has turn-level evidence (LongMemEval has_answer).
`--ids` restricts to the question ids listed in a file (e.g. the old n=30).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

METRICS = ("any_hit_at_k", "recall_at_k", "evidence_coverage_at_k", "mrr")
SHORT = {"any_hit_at_k": "hit@k", "recall_at_k": "recall@k", "evidence_coverage_at_k": "evid@k", "mrr": "MRR"}


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if len(x) == 1 or np.all(x == x[0]):
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def value(r: dict, m: str):
    v = r.get(m)
    if m == "any_hit_at_k":
        return None if v is None else (1.0 if v else 0.0)
    return None if v is None else float(v)


def scored_rows(rows: list[dict], ids: set[str] | None = None) -> dict[str, dict[str, dict]]:
    """adapter -> question_id -> row, scored rows only."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if "error" in r or not r.get("gold"):
            continue
        if ids is not None and r["question_id"] not in ids:
            continue
        out[r["adapter"]][r["question_id"]] = r
    return out


def summarize(rs: list[dict], n_boot: int, seed: int) -> dict:
    s = {"n": len(rs)}
    for m in METRICS:
        vals = [v for v in (value(r, m) for r in rs) if v is not None]
        if not vals:
            s[m] = None
            continue
        mean = sum(vals) / len(vals)
        lo, hi = wilson(int(round(sum(vals))), len(vals)) if m == "any_hit_at_k" else boot_ci(vals, n_boot, seed)
        s[m] = {"mean": mean, "lo": lo, "hi": hi, "n": len(vals)}
    return s


def paired(a: dict[str, dict], b: dict[str, dict], m: str, n_boot: int, seed: int) -> dict | None:
    common = [q for q in a if q in b and value(a[q], m) is not None and value(b[q], m) is not None]
    if not common:
        return None
    d = [value(a[q], m) - value(b[q], m) for q in common]
    lo, hi = boot_ci(d, n_boot, seed)
    return {"n": len(common), "diff": sum(d) / len(d), "lo": lo, "hi": hi,
            "a_better": sum(1 for x in d if x > 1e-12), "b_better": sum(1 for x in d if x < -1e-12),
            "ties": sum(1 for x in d if abs(x) <= 1e-12)}


def analyse(rows: list[dict], pair: tuple[str, str] | None = None, ids: set[str] | None = None,
            n_boot: int = 10000, seed: int = 0) -> dict:
    by_ad = scored_rows(rows, ids)
    out = {"overall": {}, "by_type": {}, "paired": {}, "paired_by_type": {}}
    for ad, qs in sorted(by_ad.items()):
        rs = list(qs.values())
        out["overall"][ad] = summarize(rs, n_boot, seed)
        types = defaultdict(list)
        for r in rs:
            types[r.get("qtype") or "?"].append(r)
        out["by_type"][ad] = {t: summarize(trs, n_boot, seed) for t, trs in sorted(types.items())}
    if pair and pair[0] in by_ad and pair[1] in by_ad:
        A, B = by_ad[pair[0]], by_ad[pair[1]]
        out["paired"] = {m: paired(A, B, m, n_boot, seed) for m in METRICS}
        types = sorted({r.get("qtype") or "?" for r in A.values()})
        for t in types:
            At = {q: r for q, r in A.items() if (r.get("qtype") or "?") == t}
            Bt = {q: r for q, r in B.items() if (r.get("qtype") or "?") == t}
            out["paired_by_type"][t] = {m: paired(At, Bt, m, n_boot, seed) for m in METRICS}
    return out


def _fmt(c) -> str:
    if not c:
        return "n/a"
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def _fmt_d(p) -> str:
    if not p:
        return "n/a"
    sig = " *" if (p["lo"] > 0 or p["hi"] < 0) else ""
    return f"{p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]{sig} ({p['a_better']}/{p['b_better']}/{p['ties']})"


def render(res: dict, pair: tuple[str, str] | None = None) -> str:
    L = ["| adapter | n | hit@k | recall@k | evid@k | MRR |", "|---|---|---|---|---|---|"]
    for ad, s in res["overall"].items():
        L.append(f"| {ad} | {s['n']} | " + " | ".join(_fmt(s[m]) for m in METRICS) + " |")
    ads = list(res["by_type"])
    types = sorted({t for ad in ads for t in res["by_type"][ad]})
    for m in ("any_hit_at_k", "evidence_coverage_at_k", "recall_at_k"):
        L += ["", f"{SHORT[m]} by type (95% CI):", "", "| type | " + " | ".join(ads) + " |", "|---|" + "---|" * len(ads)]
        for t in types:
            cells = []
            for ad in ads:
                s = res["by_type"][ad].get(t)
                cells.append(f"{_fmt(s[m])} n={s['n']}" if s and s.get(m) else "n/a")
            L.append(f"| {t} | " + " | ".join(cells) + " |")
    if pair and res["paired"]:
        L += ["", f"paired {pair[0]} − {pair[1]} (mean diff [95% CI], * = excludes 0; {pair[0]} better / worse / tie):", "",
              "| scope | " + " | ".join(SHORT[m] for m in METRICS) + " |", "|---|" + "---|" * len(METRICS),
              "| all | " + " | ".join(_fmt_d(res["paired"][m]) for m in METRICS) + " |"]
        for t, pm in res["paired_by_type"].items():
            L.append(f"| {t} | " + " | ".join(_fmt_d(pm[m]) for m in METRICS) + " |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--pair", default=None, help="A,B: paired differences A - B")
    ap.add_argument("--ids", default=None, help="file of question ids to restrict to")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(os.path.join(a.run, "results.jsonl"), encoding="utf-8") if l.strip()]
    ids = {l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()} if a.ids else None
    pair = tuple(a.pair.split(",")) if a.pair else None
    res = analyse(rows, pair, ids, a.boot, a.seed)
    print(render(res, pair))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"run": os.path.basename(os.path.normpath(a.run)), "pair": pair, "ids": a.ids,
                       "boot": a.boot, "seed": a.seed, **res}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
