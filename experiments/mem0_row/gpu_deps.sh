#!/usr/bin/env bash
# Install the calibration's python deps on the BMA (no sudo -> --user).
# torch is pinned to the CPU wheel on purpose: it is only here to satisfy
# sentence-transformers for the 384-d embed server, the GPU belongs to
# qwen2.5:7b, and the CUDA wheel is ~2.5 GB of billed download for nothing.
set -u
export PATH="$HOME/.local/bin:$HOME/ollama/bin:$PATH"
P=python3

echo "== torch (cpu) =="
$P -m pip install --quiet --user torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2
echo "== mem0ai + deps =="
$P -m pip install --quiet --user mem0ai qdrant-client sentence-transformers 2>&1 | tail -2
echo "== versions =="
$P -c "import mem0, qdrant_client, sentence_transformers, torch; print(' mem0', mem0.__version__, '| qdrant', qdrant_client.__version__, '| st', sentence_transformers.__version__, '| torch', torch.__version__)"

echo "== embed server on :11437 =="
cd "$HOME/kannaka-bench" || exit 1
pgrep -f "bench.embed_server" >/dev/null || nohup $P -m bench.embed_server --port 11437 > "$HOME/embed.log" 2>&1 &
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:11437/v1/models >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://127.0.0.1:11437/v1/models >/dev/null 2>&1 && echo " embed server up" || { echo " EMBED SERVER DOWN"; tail -5 "$HOME/embed.log"; exit 1; }
echo "DEPS OK"
