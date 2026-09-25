"""Split the standard 30 questions (longmemeval_s, --limit 5) into N worker files, balanced by turn count (LPT)."""
import sys
sys.path.insert(0, "/home/nick/src/kannaka-bench-mem0")
from bench.datasets import longmemeval as L
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
qs, meta = L.load("longmemeval_s", limit=5)
assert len(qs) == 30, len(qs)
bins = [[0, []] for _ in range(n)]
for q in sorted(qs, key=lambda q: -len(q.items)):
    b = min(bins, key=lambda b: b[0]); b[0] += len(q.items); b[1].append(q.id)
for i, (tot, ids) in enumerate(bins):
    open(f"/home/nick/mem0row/q{i}.txt", "w").write("\n".join(ids) + "\n")
    print(i, tot, len(ids))
print("total turns", sum(len(q.items) for q in qs), "sha", meta.get("sha256", "")[:12])
