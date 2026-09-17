"""Phase 2 — answer accuracy. For every (question, adapter) row of a finished
retrieval run, give an answer model ONLY the top-k recalled items, get an
answer, and have a judge model score it against the gold answer (the
LongMemEval judge protocol: correct / incorrect given the reference).

  python -m bench.answer --run results/<run>/ [--k 5] [--adapters kannaka_minilm,vector_numpy]
      [--full-context recency]   # the plain-context baseline: the WHOLE history in the window

Models are any OpenAI-compatible endpoint (the KAX gateway on the lab hosts):
BENCH_LLM_URL (e.g. http://172.18.0.1:4000/v1), BENCH_LLM_KEY, BENCH_ANSWER_MODEL
(default agent-brain), BENCH_JUDGE_MODEL (default = answer model). Writes
<run>/answers.jsonl (one row per question per adapter: answer, verdict,
tokens) and prints accuracy per adapter and per question type. Re-runs skip
rows already judged. Every call's token counts are recorded — cost is a
column, not a footnote.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict

FULL_CONTEXT_CHAR_CAP = int(os.environ.get("BENCH_FULL_CONTEXT_CHARS", "400000"))

ANSWER_SYS = ("You are answering a question about a user's past conversations with an assistant. You are given "
              "excerpts retrieved from those conversations (they may be irrelevant or incomplete) and the date the "
              "question is asked. Answer briefly and concretely from the excerpts. If the excerpts do not contain the "
              "answer, say \"I don't know\".")
JUDGE_SYS = ("You are grading an answer to a question against a reference answer. Reply with exactly one word: "
             "CORRECT if the answer conveys the same fact(s) as the reference (wording may differ; extra correct "
             "detail is fine), otherwise INCORRECT. An answer of \"I don't know\" is INCORRECT unless the reference "
             "says the information is unavailable.")


def chat(url: str, key: str, model: str, system: str, user: str, max_tokens: int = 300, temperature: float = 0.0) -> tuple[str, dict]:
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = d.get("usage") or {}
            return text.strip(), {"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens")}
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return "", {}


def load_questions(dataset: str, limit):
    if dataset == "locomo":
        from .datasets import locomo
        convs, _ = locomo.load(limit_questions=limit)
        out = {}
        for _cid, items, qs in convs:
            by_id = {it.id: it for it in items}
            for q in qs:
                out[q.id] = (q, by_id, items)
        return out
    from .datasets import longmemeval
    qs, _ = longmemeval.load(dataset, limit=limit)
    return {q.id: (q, {it.id: it for it in q.items}, q.items) for q in qs}


def format_excerpts(items, gold_level: str) -> str:
    lines = []
    for it in items:
        when = it.when.strftime("%Y-%m-%d") if it.when else "undated"
        lines.append(f"[{when}] {it.text[:1500]}")
    return "\n\n".join(lines) if lines else "(nothing retrieved)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=None, help="use the top-k of the run's hits (default: the run's k)")
    ap.add_argument("--adapters", default=None, help="comma list; default all in the run")
    ap.add_argument("--full-context", default="", help="adapter name(s) whose answer sees the WHOLE history instead of hits")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    url = os.environ.get("BENCH_LLM_URL", "http://127.0.0.1:4000/v1")
    key = os.environ.get("BENCH_LLM_KEY", "")
    answer_model = os.environ.get("BENCH_ANSWER_MODEL", "agent-brain")
    judge_model = os.environ.get("BENCH_JUDGE_MODEL", answer_model)
    if not key:
        sys.exit("BENCH_LLM_KEY not set")

    manifest = json.load(open(os.path.join(a.run, "manifest.json"), encoding="utf-8"))
    dataset = manifest["dataset"]["dataset"]
    limit = manifest.get("limit")
    rows = [json.loads(l) for l in open(os.path.join(a.run, "results.jsonl"), encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "error" not in r]
    if a.adapters:
        keep = set(a.adapters.split(","))
        rows = [r for r in rows if r["adapter"] in keep]
    full = set(x for x in a.full_context.split(",") if x)
    qmap = load_questions("locomo" if dataset.startswith("locomo") else dataset, limit)

    out_path = os.path.join(a.run, "answers.jsonl")
    done = {}
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                done[(d["adapter"], d["question_id"])] = d
    results = list(done.values())
    n_new = 0
    for r in rows:
        if (r["adapter"], r["question_id"]) in done:
            continue
        if r["question_id"] not in qmap:
            continue
        q, by_id, items = qmap[r["question_id"]]
        k = a.k or r["k"]
        if r["adapter"] in full:
            ctx_items = items
            text = format_excerpts(ctx_items, r["gold_level"])
            if len(text) > FULL_CONTEXT_CHAR_CAP:
                text = text[-FULL_CONTEXT_CHAR_CAP:]
            mode = "full-context"
        else:
            ctx_items = [by_id[h] for h in r["hits"][:k] if h in by_id]
            text = format_excerpts(ctx_items, r["gold_level"])
            mode = f"top-{k}"
        asked = q.asked_at.strftime("%Y-%m-%d") if q.asked_at else "unknown date"
        user = f"Question date: {asked}\n\nExcerpts:\n{text}\n\nQuestion: {q.question}"
        t0 = time.perf_counter()
        answer, au = chat(url, key, answer_model, ANSWER_SYS, user, max_tokens=200)
        ans_ms = (time.perf_counter() - t0) * 1000
        verdict, ju = chat(url, key, judge_model, JUDGE_SYS,
                           f"Question: {q.question}\nReference answer: {q.answer}\nAnswer to grade: {answer}\n\nOne word: CORRECT or INCORRECT.",
                           max_tokens=5)
        correct = verdict.strip().upper().startswith("CORRECT")
        row = {"adapter": r["adapter"], "question_id": r["question_id"], "qtype": r["qtype"], "mode": mode,
               "hit_at_k": r["any_hit_at_k"], "answer": answer, "gold": q.answer, "verdict": verdict, "correct": correct,
               "answer_model": answer_model, "judge_model": judge_model, "answer_ms": round(ans_ms),
               "tokens": {"answer": au, "judge": ju}}
        results.append(row)
        n_new += 1
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{r['adapter']:14}] {r['question_id'][:12]} {r['qtype'][:20]:20} {mode:8} hit={r['any_hit_at_k']!s:5} "
              f"-> {'CORRECT' if correct else 'wrong  '} | {answer[:70]!r}", flush=True)

    print(f"\n{n_new} new rows; {len(results)} total in {out_path}\n")
    by = defaultdict(list)
    for x in results:
        by[x["adapter"]].append(x)
    print("| adapter | n | accuracy | acc when hit | acc when miss | prompt tok/q | answer p50 ms |")
    print("|---|---|---|---|---|---|---|")
    for name, xs in sorted(by.items(), key=lambda kv: -sum(x["correct"] for x in kv[1]) / max(1, len(kv[1]))):
        acc = sum(x["correct"] for x in xs) / len(xs)
        hit = [x for x in xs if x["hit_at_k"]]
        miss = [x for x in xs if not x["hit_at_k"]]
        acc_h = sum(x["correct"] for x in hit) / len(hit) if hit else float("nan")
        acc_m = sum(x["correct"] for x in miss) / len(miss) if miss else float("nan")
        ptok = sum((x["tokens"]["answer"] or {}).get("prompt") or 0 for x in xs) / len(xs)
        lat = sorted(x["answer_ms"] for x in xs)[len(xs) // 2]
        print(f"| {name} | {len(xs)} | {acc:.3f} | {acc_h:.3f} | {acc_m:.3f} | {ptok:.0f} | {lat} |")
    types = sorted({x["qtype"] for x in results})
    if types:
        print("\naccuracy by type:\n")
        print("| type | " + " | ".join(by.keys()) + " |")
        print("|---|" + "---|" * len(by))
        for t in types:
            cells = []
            for name in by:
                xs = [x for x in by[name] if x["qtype"] == t]
                cells.append(f"{sum(x['correct'] for x in xs) / len(xs):.2f} (n={len(xs)})" if xs else "—")
            print(f"| {t} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
