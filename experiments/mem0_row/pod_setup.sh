#!/usr/bin/env bash
# On the BMA: ollama + qwen2.5:7b (extraction) + qwen2.5:14b (answer/judge, pulled now, used after ingest),
# python deps, embed server (same all-MiniLM as every row), the dataset.
set -u
cd "$HOME"
bash "$HOME/gpu_bootstrap.sh" || exit 1
export PATH="$HOME/.local/bin:$HOME/ollama/bin:$PATH"
bash "$HOME/gpu_deps.sh" || exit 1
echo "== dataset =="
python3 -c "
import sys; sys.path.insert(0,'$HOME/kannaka-bench')
from bench.datasets import longmemeval as L
qs,m=L.load('longmemeval_s',limit=5); print(len(qs), sum(len(q.items) for q in qs), m.get('sha256','')[:12])
"
echo "== versions =="
python3 -c "import mem0, qdrant_client; print('mem0', mem0.__version__, 'qdrant-client', qdrant_client.__version__)"
ollama --version | head -1
python3 -m site --user-site > "$HOME/user_site"
echo "== pull qwen2.5:14b (background) =="
nohup ollama pull qwen2.5:14b > "$HOME/pull14b.log" 2>&1 &
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "SETUP OK"
