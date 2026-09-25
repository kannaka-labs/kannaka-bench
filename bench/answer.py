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
import re
import sys
import time
import urllib.request
from collections import defaultdict

# The plain-context baseline sends the whole history. At 400k chars that was
# ~94k prompt tokens per question (2026-09-17) — five such rows cost more than
# the other forty combined and, on a low balance, ended the run. 120k chars is
# ~30k tokens: still far beyond any retrieval window, honest as a "give the
# model everything" floor, and a cost a run can afford to repeat.
FULL_CONTEXT_CHAR_CAP = int(os.environ.get("BENCH_FULL_CONTEXT_CHARS", "120000"))


class LlmError(RuntimeError):
    pass


# Phase 2 v1 (2026-09-17) cut every excerpt at 1500 chars: 28% of LongMemEval
# turns are longer (p90 2556) and the answer model said "cut off" out loud.
EXCERPT_CHAR_CAP = int(os.environ.get("BENCH_EXCERPT_CHARS", "6000"))
# Each retrieved turn is shown with its neighbour (the assistant reply that
# follows a user turn, or the user turn before an assistant one): many gold
# answers live in the reply to the retrieved turn ("what did you recommend?").
PAIR_TURNS = os.environ.get("BENCH_PAIR_TURNS", "1") not in ("0", "false", "no")

ANSWER_SYS = ("You are answering a question about a user's past conversations with an assistant. You are given "
              "excerpts retrieved from those conversations, in chronological order, each with its date, and the "
              "date the question is asked. They may be irrelevant or incomplete. Answer briefly and concretely from "
              "the excerpts. If two excerpts state conflicting facts, the most recent one is current. If the "
              "excerpts do not contain the answer, say \"I don't know\".")
ANSWER_SYS_PREFERENCE = ("You are answering a question about a user's past conversations with an assistant. You are "
                         "given excerpts retrieved from those conversations, in chronological order with dates. The "
                         "user is asking for a suggestion or recommendation: answer it, and make the answer fit what "
                         "the excerpts show about the user's situation, tastes and constraints, naming those specifics. "
                         "Do not say \"I don't know\": if the excerpts show nothing relevant, give a generic answer.")
ANSWER_SYS_AGGREGATE = (
    "You are answering a question about a user's past conversations with an assistant. You are given excerpts "
    "retrieved from those conversations, in chronological order, each with its date, and the date the question is "
    "asked. They may be irrelevant or incomplete. If two excerpts state conflicting facts about the SAME thing, the "
    "most recent one is current.\n"
    "This question asks for a number — a count, a total, or an elapsed time — so answer it in two steps.\n"
    "Step 1 - enumerate. List every item, event or value in the excerpts that could qualify, each with its date. "
    "Sweep all of them: what one question needs is usually spread across several conversations on different dates, "
    "and an item mentioned only in passing still counts. Do not drop one on your own reasoning unless the excerpts "
    "explicitly rule it out.\n"
    "Step 2 - compute. Apply what the question actually asks to that list and work the number out. Counting a set of "
    "things means counting the distinct ones. A combined time, distance or amount means ADDING the individual values "
    "- naming the parts separately is not an answer. Time between two dates means subtracting them.\n"
    "If the excerpts genuinely do not contain what is needed, say \"I don't know\". Otherwise end with the final "
    "answer on its own last line, as a plain number or short phrase."
)

JUDGE_SYS = ("You are grading an answer to a question against a reference answer. Reply with exactly one word: "
             "CORRECT if the answer conveys the same fact(s) as the reference (wording may differ; extra correct "
             "detail is fine), otherwise INCORRECT. An answer of \"I don't know\" is INCORRECT unless the reference "
             "says the information is unavailable.")



# Routing reads the QUESTION, never `qtype` — the dataset's gold label, which
# a deployed system does not have. Deliberately narrow and auditable: every
# row records the route it took, so a regression can be attributed to routing
# rather than to the prompt it selected.
# Every LongMemEval question needing a computed number opens with one of
# these; tuned against the 30 real question texts, not invented ones. It fires
# on multi-session (5/5) and also on the temporal and knowledge-update rows
# that ask for a number — deliberately, because the prompt it selects is a
# superset of the plain one and those categories need the same arithmetic.
AGGREGATE_RE = re.compile(
    r"\b(how many|how much|how long|how often|in total|altogether|combined|total number)\b",
    re.I,
)
# Real preference rows are all "Can you recommend/suggest ...". The trap is
# single-session-assistant: "Can you remind me ... the restaurant you
# recommended" is a past-fact lookup that scores 1.00, and the preference
# prompt forbids "I don't know" and asks for a generic answer — so matching
# bare "recommend" would replace a correct recall with an invention.
PREFERENCE_RE = re.compile(
    r"\b(can you (?:recommend|suggest)|could you (?:recommend|suggest)"
    r"|what should i|which should i|any (?:ideas|suggestions|recommendations)"
    r"|do you have any (?:suggestions|recommendations))\b",
    re.I,
)
REMINDER_RE = re.compile(r"\bremind me\b|\bremember\b|\bwhat was\b|\bwhat did\b", re.I)


def route_question(question: str, qtype: str, mode: str) -> str:
    """Which system prompt to use: 'aggregate', 'preference' or 'plain'.

    `mode='qtype'` reproduces the pre-v4 behaviour (gold label) so the cost of
    the leak is measurable; `mode='question'` is the honest, deployable path.
    """
    if mode == "qtype":
        return "preference" if "preference" in (qtype or "") else "plain"
    q = question or ""
    if AGGREGATE_RE.search(q):
        return "aggregate"
    # A question asking what was said or recommended IN THE PAST is a lookup,
    # whatever verbs it contains — it must never reach the preference prompt.
    if PREFERENCE_RE.search(q) and not REMINDER_RE.search(q):
        return "preference"
    return "plain"


SYS_BY_ROUTE = {
    "aggregate": ANSWER_SYS_AGGREGATE,
    "preference": ANSWER_SYS_PREFERENCE,
    "plain": ANSWER_SYS,
}


# The judge is asked to mark an abstention INCORRECT (unless the reference
# itself says the information is unavailable) and does not always comply — it
# graded two near-identical abstentions on the same question differently for
# two adapters. A judge that lets abstentions through rewards the systems that
# answer least, so the rule is enforced here instead of only requested.
ABSTAIN_RE = re.compile(
    r"(i don'?t know"
    r"|do(?:es)? not (?:contain|specify|mention|provide|state|include)"
    r"|don'?t (?:contain|specify|mention|provide|state)"
    r"|cannot (?:determine|calculate|tell|find|answer)"
    r"|can'?t (?:determine|calculate|tell|find)"
    r"|not enough information"
    r"|unable to determine"
    r"|no (?:specific )?(?:information|mention|record|details))",
    re.I,
)
# A gold answer that itself says the information is not available — the one
# case where abstaining IS the correct answer.
GOLD_UNAVAILABLE_RE = re.compile(
    r"(not (?:available|mentioned|specified|stated|provided)"
    r"|no (?:information|mention|record)"
    r"|unavailable|did not (?:say|mention|specify))",
    re.I,
)


def is_abstention(answer: str) -> bool:
    """True when the answer's FINAL position is 'I cannot say'.

    Deliberately narrow: the last non-empty line, or a short whole answer.
    An answer that hedges in the middle and then commits to a value is not an
    abstention, and must not be scored as one.
    """
    a = (answer or "").strip()
    if not a:
        return True
    lines = [l.strip() for l in a.splitlines() if l.strip()]
    if lines and ABSTAIN_RE.search(lines[-1]):
        return True
    return len(a) < 200 and bool(ABSTAIN_RE.search(a))


def grade(verdict: str, answer: str, gold: str) -> tuple[bool, bool]:
    """(correct, overridden). Enforces the abstention rule the judge is given."""
    correct = verdict.strip().upper().startswith("CORRECT")
    if correct and is_abstention(answer) and not GOLD_UNAVAILABLE_RE.search(gold or ""):
        return False, True
    return correct, False

def chat(url: str, key: str, model: str, system: str, user: str, max_tokens: int = 300, temperature: float = 0.0) -> tuple[str, dict]:
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = d.get("usage") or {}
            return text.strip(), {"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens")}
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            last = LlmError(f"HTTP {e.code}: {body}")
            # 4xx other than rate limiting will not get better by retrying
            if e.code < 500 and e.code != 429:
                raise last
        except Exception as e:  # noqa: BLE001
            last = LlmError(f"{type(e).__name__}: {str(e)[:200]}")
        time.sleep(2 * (attempt + 1))
    raise last or LlmError("no response")


def load_questions(dataset: str, limit, manifest=None):
    if dataset.startswith("convomem"):
        from .datasets import convomem
        ctx = int((manifest or {}).get("dataset", {}).get("context_size") or 50)
        qs, _ = convomem.load(context=ctx, limit=limit)
        return {q.id: (q, {it.id: it for it in q.items}, q.items) for q in qs}
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


def format_excerpts(items, gold_level: str, cap: int | None = None) -> str:
    cap = cap or EXCERPT_CHAR_CAP
    lines = []
    for it in items:
        when = it.when.strftime("%Y-%m-%d") if it.when else "undated"
        lines.append(f"[{when}] {it.text[:cap]}")
    return "\n\n".join(lines) if lines else "(nothing retrieved)"


def _turn_key(item_id: str):
    sid, _, t = item_id.rpartition("#")
    return (sid, int(t)) if sid and t.isdigit() else (item_id, 0)


def with_siblings(hit_ids, candidates, per_session: int, max_excerpts: int):
    """Hits plus up to per_session-1 further turns of each hit's session taken
    from the deeper candidate list in rank order; sessions keep the hits'
    order; the result is cut to max_excerpts (before pair expansion)."""
    if per_session <= 1 or not candidates:
        return hit_ids[:max_excerpts]
    by_session: dict = {}
    for c in candidates:
        by_session.setdefault(c.rpartition("#")[0] if "#" in c else c, []).append(c)
    out, seen = [], set()
    for h in hit_ids:
        sid = h.rpartition("#")[0] if "#" in h else h
        picked = [h] + [c for c in by_session.get(sid, []) if c != h][: per_session - 1]
        for c in picked:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out[:max_excerpts]


def expand_pairs(hit_ids, by_id):
    """The hits plus each one's conversational partner turn, deduplicated and
    in history order (LongMemEval ids are <session>#<turn>; other datasets
    keep the hits as they are)."""
    keep = []
    seen = set()
    for h in hit_ids:
        if h not in by_id:
            continue
        sid, _, t = h.rpartition("#")
        cands = [h]
        if PAIR_TURNS and sid and t.isdigit():
            ti = int(t)
            me = by_id[h]
            partner = f"{sid}#{ti + 1}" if me.text.startswith("user:") else f"{sid}#{ti - 1}"
            if partner in by_id:
                cands.append(partner)
        for c in cands:
            if c not in seen:
                seen.add(c)
                keep.append(by_id[c])
    keep.sort(key=lambda it: ((it.when or __import__("datetime").datetime.min.replace(tzinfo=__import__("datetime").timezone.utc)), _turn_key(it.id)))
    return keep


def native_items(row, k: int):
    """What a rewriting adapter (Mem0) itself returned: its stored memory text,
    dated where it kept the date, in chronological order like every other
    excerpt list. None when the row carries no text (an older run, or an
    adapter that stores the dataset's own items)."""
    from datetime import datetime
    from types import SimpleNamespace
    texts = row.get("hit_texts")
    if texts is None:
        return None
    whens = row.get("hit_whens") or [None] * len(texts)
    out = []
    for i, (t, w) in enumerate(zip(texts[:k], whens[:k])):
        dt = None
        if w:
            try:
                dt = datetime.fromisoformat(w)
            except ValueError:
                dt = None
        out.append(SimpleNamespace(id=f"native#{i}", text=t, when=dt))
    far = __import__("datetime").datetime.min.replace(tzinfo=__import__("datetime").timezone.utc)
    out.sort(key=lambda it: (it.when or far, _turn_key(it.id)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=None, help="use the top-k of the run's hits (default: the run's k)")
    ap.add_argument("--adapters", default=None, help="comma list; default all in the run")
    ap.add_argument("--full-context", default="", help="adapter name(s) whose answer sees the WHOLE history instead of hits")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="write answers-<tag>.jsonl instead of answers.jsonl")
    ap.add_argument("--max-answer-tokens", type=int, default=600,
                    help="answer model output cap (ConvoMem gold answers are multi-item lists; 200 cut them off mid-list)")
    ap.add_argument("--per-session", type=int, default=1,
                    help="turns per hit session shown to the model, filled from the row's candidates (default 1)")
    ap.add_argument("--max-excerpts", type=int, default=40, help="cap on excerpts before pair expansion")
    ap.add_argument("--native-text", action="store_true",
                    help="for rows that carry hit_texts (Mem0), answer from the adapter's OWN returned "
                         "memory text instead of the dataset turns its ids point to; rows without it are skipped")
    ap.add_argument("--route", choices=("question", "qtype"), default="question",
                    help="how to pick the system prompt: from the question text (default, deployable) "
                         "or from the dataset's gold qtype label (pre-v4 behaviour, for comparison)")
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
    qmap = load_questions("locomo" if dataset.startswith("locomo") else dataset, limit, manifest)

    out_path = os.path.join(a.run, f"answers-{a.tag}.jsonl" if a.tag else "answers.jsonl")
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
        native = native_items(r, k) if a.native_text else None
        if a.native_text and native is None:
            continue
        if native is not None:
            ctx_items = native
            text = format_excerpts(ctx_items, r["gold_level"])
            mode = f"native-top-{k}"
        elif r["adapter"] in full:
            ctx_items = items
            text = format_excerpts(ctx_items, r["gold_level"])
            if len(text) > FULL_CONTEXT_CHAR_CAP:
                text = text[-FULL_CONTEXT_CHAR_CAP:]
            mode = "full-context"
        else:
            ctx_items = expand_pairs(with_siblings(r["hits"][:k], r.get("candidates") or [], a.per_session, a.max_excerpts), by_id)
            text = format_excerpts(ctx_items, r["gold_level"])
            mode = f"top-{k}"
        asked = q.asked_at.strftime("%Y-%m-%d") if q.asked_at else "unknown date"
        user = f"Question date: {asked}\n\nExcerpts:\n{text}\n\nQuestion: {q.question}"
        route = route_question(q.question, r.get("qtype") or "", a.route)
        system = SYS_BY_ROUTE[route]
        t0 = time.perf_counter()
        try:
            answer, au = chat(url, key, answer_model, system, user, max_tokens=a.max_answer_tokens)
            ans_ms = (time.perf_counter() - t0) * 1000
            verdict, ju = chat(url, key, judge_model, JUDGE_SYS,
                               f"Question: {q.question}\nReference answer: {q.answer}\nAnswer to grade: {answer}\n\nOne word: CORRECT or INCORRECT.",
                               max_tokens=5)
        except LlmError as e:
            # Not graded: recorded as an error row (not counted as wrong), and the
            # run keeps going — but an out-of-credit or auth failure will hit every
            # row, so stop after a few in a row rather than burn the whole run.
            print(f"[{r['adapter']:14}] {r['question_id'][:12]} LLM ERROR: {e}", flush=True)
            consecutive_errors = globals().get("_consec", 0) + 1
            globals()["_consec"] = consecutive_errors
            if consecutive_errors >= 3:
                print("three consecutive LLM errors — stopping; nothing below was graded", flush=True)
                break
            continue
        globals()["_consec"] = 0
        correct, judge_overridden = grade(verdict, answer, q.answer)
        row = {"adapter": r["adapter"], "question_id": r["question_id"], "qtype": r["qtype"], "mode": mode,
               "n_excerpts": len(ctx_items), "excerpt_cap": EXCERPT_CHAR_CAP, "pair_turns": PAIR_TURNS,
               "per_session": a.per_session, "max_excerpts": a.max_excerpts, "max_answer_tokens": a.max_answer_tokens,
               "route": route, "route_mode": a.route,
               "judge_overridden": judge_overridden,
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
