"""Merge the 6 worker runs into one run dir, the shape every other published run has."""
import glob, json, os, re
src = os.path.expanduser("~/mem0row/pod/results")
dst = os.path.expanduser("~/kannaka-bench-results/s-5pertype-k15-mem0")
os.makedirs(dst, exist_ok=True)
rows, mans = [], []
for d in sorted(glob.glob(os.path.join(src, "mem0-w*"))):
    rows += [json.loads(l) for l in open(os.path.join(d, "results.jsonl"), encoding="utf-8") if l.strip()]
    mp = os.path.join(d, "manifest.json")
    if os.path.exists(mp):
        mans.append(json.load(open(mp)))
ok = [r for r in rows if "error" not in r]
std = set(open(os.path.expanduser("~/mem0row/std30.txt")).read().split())
got = {r["question_id"] for r in ok}
print(f"rows {len(rows)}, ok {len(ok)}, errors {len(rows) - len(ok)}, standard ids covered {len(got & std)}/30, extra {len(got - std)}")
m = dict(mans[0]) if mans else {}
m["run_id"] = "s-5pertype-k15-mem0"
m["workers"] = [{"run_id": x["run_id"], "started_at": x["started_at"], "finished_at": x.get("finished_at"),
                 "wall_s": x.get("wall_s"), "stores": x.get("stores")} for x in mans]
m["started_at"] = min(x["started_at"] for x in mans) if mans else None
m["finished_at"] = max((x.get("finished_at") or "") for x in mans) if mans else None
m["stores"] = sum(x.get("stores", 0) for x in mans)
tot = {}
for r in ok:
    for k, v in (r.get("ingest_stats") or {}).items():
        tot[k] = tot.get(k, 0) + v
m["ingest_stats"] = {"mem0": tot}
sp = os.path.expanduser("~/mem0row/setup.json")
m["mem0_setup"] = json.load(open(sp)) if os.path.exists(sp) else {}
# Real LLM requests in the scored window, from ollama's access log (retries included).
ol = os.path.expanduser("~/mem0row/pod/ollama.log")
ts_path = os.path.expanduser("~/mem0row/pod/launch_ts")
if os.path.exists(ol) and os.path.exists(ts_path):
    since = open(ts_path).read().strip()          # "YYYY/MM/DD - HH:MM:SS", the scored launch
    end = (m.get("finished_at") or "")[:19].replace("-", "/").replace("T", " - ")
    ok200 = re.compile(r"\|\s*200\s*\|")
    lines = [l for l in open(ol, encoding="utf-8", errors="replace")
             if '"/v1/chat/completions"' in l and l[6:27] >= since and (not end or l[6:27] <= end)]
    m["ollama_chat_requests"] = len(lines)
    m["ollama_chat_non200"] = sum(1 for l in lines if not ok200.search(l))
    m["ollama_window"] = [since, end]
json.dump(m, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
with open(os.path.join(dst, "results.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps({k: m.get(k) for k in ("stores", "ingest_stats", "ollama_chat_requests", "ollama_chat_non200",
                                         "ollama_window", "started_at", "finished_at")}, indent=1))
