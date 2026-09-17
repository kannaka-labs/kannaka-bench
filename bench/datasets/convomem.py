"""ConvoMem (Salesforce AI Research, 2025; HF `Salesforce/ConvoMem`, CC-BY-NC-4.0 —
non-commercial, attribution): conversational-memory questions over pre-mixed
haystacks of user/assistant conversations at fixed context sizes (1 … 300
conversations). Six categories: user_evidence, assistant_facts_evidence,
changing_evidence, preference_evidence, implicit_connection_evidence,
abstention_evidence (gold: no information exists — retrieval metrics are
undefined there; the answer stage scores the refusal).

Layout on the hub: core_benchmark/pre_mixed_testcases/<category>/<n>_evidence/
batched_NNN.json — each file is a list of test cases, one contextSize per
file (batch index rises with context size), each case:
  {"evidenceItems": [{"question", "answer", "message_evidences": [{"speaker",
   "text"}], "conversations": [{"id", ...}], "category", "personId", ...}],
   "conversations": [{"id", "messages": [{"speaker", "text"}], "containsEvidence"}],
   "contextSize": "N"}

We ingest one item per message, id "<conversation id>#<index>", text
"<speaker>: <text>", no timestamps (the dataset carries none; conversation
order is the only history order). Gold = the evidence conversation ids
(session level), so hit/recall are comparable with LongMemEval's rows.

  load(context=50, limit=5, categories=None, n_evidence=None)
    -> (questions, meta); `limit` is per category (stratified), `n_evidence`
    restricts to k-evidence subfolders (default: all present).

Files are fetched on demand into KANNAKA_BENCH_DATA/convomem/ and a small
index (file -> contextSize) is cached so later loads open only what they need.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

from ..adapters.base import MemoryItem
from .longmemeval import DATA_DIR, Question

HF = "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/core_benchmark/pre_mixed_testcases/"
CATEGORIES = ["user_evidence", "assistant_facts_evidence", "changing_evidence", "preference_evidence",
              "implicit_connection_evidence", "abstention_evidence"]
API = "https://huggingface.co/api/datasets/Salesforce/ConvoMem"
LOCAL = os.path.join(DATA_DIR, "convomem")
INDEX = os.path.join(LOCAL, "index.json")
LISTING = os.path.join(LOCAL, "listing.json")


def listing() -> list[str]:
    """The hub's file list under pre_mixed_testcases (cached): the source of
    truth for which <category>/<n>_evidence/batched_NNN.json files exist."""
    try:
        return json.load(open(LISTING, encoding="utf-8"))
    except (OSError, ValueError):
        pass
    with urllib.request.urlopen(API, timeout=60) as r:
        d = json.load(r)
    prefix = "core_benchmark/pre_mixed_testcases/"
    files = sorted(s["rfilename"][len(prefix):] for s in d.get("siblings", [])
                   if s.get("rfilename", "").startswith(prefix) and s["rfilename"].endswith(".json"))
    os.makedirs(LOCAL, exist_ok=True)
    json.dump(files, open(LISTING, "w", encoding="utf-8"))
    return files


def _index():
    try:
        return json.load(open(INDEX, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_index(ix):
    os.makedirs(LOCAL, exist_ok=True)
    tmp = INDEX + ".tmp"
    json.dump(ix, open(tmp, "w", encoding="utf-8"), indent=0)
    os.replace(tmp, INDEX)


def fetch(rel: str) -> str:
    """Download one batched file (≈6 MB) if missing; returns the local path."""
    path = os.path.join(LOCAL, rel.replace("/", os.sep))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(HF + rel, tmp)
        os.replace(tmp, path)
    return path


def _sizes_in(rel: str, ix: dict) -> list[str]:
    if rel in ix:
        return ix[rel]
    data = json.load(open(fetch(rel), encoding="utf-8"))
    sizes = sorted({str(x.get("contextSize")) for x in data}, key=lambda s: int(s) if s.isdigit() else 0)
    ix[rel] = sizes
    _save_index(ix)
    return sizes


def _files(category: str, n_evidence=None):
    """(n_evidence, relative path) for the category's batched files, in
    (n, batch index) order — the order context size rises in."""
    for rel in listing():
        parts = rel.split("/")
        if len(parts) != 3 or parts[0] != category or not parts[1].endswith("_evidence"):
            continue
        try:
            n = int(parts[1].split("_", 1)[0])
        except ValueError:
            continue
        if n_evidence and n not in n_evidence:
            continue
        yield n, rel


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _question(case: dict, category: str, n: int, rel: str, idx: int) -> Question | None:
    evs = case.get("evidenceItems") or []
    if not evs:
        return None
    ev = evs[0]
    items: list[MemoryItem] = []
    for c in case.get("conversations") or []:
        cid = c.get("id") or _sha(json.dumps(c.get("messages", [])[:2]))
        for t, m in enumerate(c.get("messages") or []):
            text = (m.get("text") or "").strip()
            if not text:
                continue
            items.append(MemoryItem(id=f"{cid}#{t}", text=f"{m.get('speaker', 'user')}: {text}", when=None,
                                    meta={"session": cid, "has_answer": False}))
    gold = {c.get("id") for c in (ev.get("conversations") or []) if c.get("id")}
    # mark the evidence messages when their text is found verbatim (a diagnostic, not the score)
    ev_texts = {(e.get("text") or "").strip() for e in (ev.get("message_evidences") or []) if isinstance(e, dict)}
    if ev_texts:
        for it in items:
            if it.text.split(": ", 1)[-1] in ev_texts:
                it.meta["has_answer"] = True
    if category == "abstention_evidence":
        gold = set()
    qid = f"convomem/{category}/{n}/{os.path.basename(rel)[:-5]}/{idx}"
    return Question(id=qid, question=ev.get("question") or "", answer=ev.get("answer") or "", qtype=category.replace("_evidence", ""),
                    asked_at=None, items=items, gold_ids=gold, gold_level="session",
                    meta={"context_size": str(case.get("contextSize")), "n_evidence": n, "person": ev.get("personId"),
                          "scenario": (ev.get("scenario_description") or "")[:200]})


def load(context: int = 50, limit: int | None = 5, categories=None, n_evidence=None):
    """Questions whose haystack has `context` conversations, `limit` per
    category. Scans batched files in order (index rises with context size),
    opening only those the cached index does not already rule out."""
    ix = _index()
    want = str(context)
    cats = categories or CATEGORIES
    n_evidence = set(n_evidence) if n_evidence else None
    out: list[Question] = []
    opened = 0
    for category in cats:
        got = 0
        files = list(_files(category, n_evidence))
        # Context size rises with the batch index and the LAST file of a subfolder
        # holds every large context (50..300) mixed — ~800 MB (user_evidence/1,
        # 2026-09-17). Scan from the end for large contexts so the first file
        # opened is the one that has them; small contexts are found at the front.
        if context >= 20:
            files = files[::-1]
        for n, rel in files:
            sizes = _sizes_in(rel, ix)
            opened += 1
            if want not in sizes:
                continue
            data = json.load(open(fetch(rel), encoding="utf-8"))
            for idx, case in enumerate(data):
                if str(case.get("contextSize")) != want:
                    continue
                q = _question(case, category, n, rel, idx)
                if q is None:
                    continue
                out.append(q)
                got += 1
                if limit and got >= limit:
                    break
            if limit and got >= limit:
                break
    meta = {"dataset": f"convomem_c{context}", "source": "Salesforce/ConvoMem (CC-BY-NC-4.0)", "context_size": context,
            "limit_per_category": limit, "categories": cats, "questions_total": len(out), "files_opened": opened}
    return out, meta
