#!/usr/bin/env bash
# Bootstrap the 4090 BMA for the mem0 extraction calibration.
# No sudo on this image, so everything lands under $HOME. Ollama now ships
# .tar.zst (the old .tgz URL 404s) and the image has no zstd binary, so the
# archive is decompressed with python's zstandard instead of tar --zstd.
set -u
cd "$HOME" || exit 1
export PATH="$HOME/ollama/bin:$PATH"

if [ ! -x "$HOME/ollama/bin/ollama" ]; then
  echo "== fetch =="
  python3 -m pip install --quiet --user zstandard 2>&1 | tail -1
  curl -fsSL -o /tmp/ollama.tar.zst \
    https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst || exit 1
  ls -l /tmp/ollama.tar.zst
  python3 -c "
import zstandard, shutil
with open('/tmp/ollama.tar.zst','rb') as f, open('/tmp/ollama.tar','wb') as g:
    zstandard.ZstdDecompressor().copy_stream(f, g)
print('decompressed')
" || exit 1
  mkdir -p "$HOME/ollama"
  tar -C "$HOME/ollama" -xf /tmp/ollama.tar || exit 1
fi
ollama --version 2>&1 | head -1

echo "== serve =="
# 8 parallel slots: the cost question is whether concurrent workers cut the
# wall clock, so concurrency has to be measurable, not just single-stream.
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
pgrep -x ollama >/dev/null || nohup ollama serve > "$HOME/ollama.log" 2>&1 &
for i in $(seq 1 40); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || { echo "OLLAMA DID NOT START"; tail -5 "$HOME/ollama.log"; exit 1; }
echo "serve up"

echo "== pull qwen2.5:7b =="
ollama pull qwen2.5:7b 2>&1 | tail -1

echo "== warm (load into VRAM; excluded from the timing) =="
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"hi","stream":false,"options":{"num_predict":8}}' >/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "BOOTSTRAP OK"
