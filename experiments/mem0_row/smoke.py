import sys, time
from bench.datasets import longmemeval as L
from bench.adapters.mem0 import Mem0Adapter
qs, _ = L.load("longmemeval_s", limit=5)
q = qs[0]
gold_items = [it for it in q.items if (it.meta or {}).get("has_answer")][:2]
items = gold_items + q.items[:4]
ad = Mem0Adapter(); ad.open("/tmp/smoke-mem0")
t0 = time.time(); ad.ingest(items); el = time.time() - t0
print("ingest", len(items), "turns in", round(el, 1), "s", ad.ingest_stats())
hits = ad.recall(q.question, 15, q.asked_at)
print("Q:", q.question, "| gold ids", [i.id for i in gold_items])
for h in hits[:6]:
    print(" ", h.id, round(h.score, 3), h.when, "|", h.text[:90])
