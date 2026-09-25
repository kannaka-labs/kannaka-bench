#!/usr/bin/env bash
# Within-model LOCAL answer pass (qwen2.5:14b on the pod's 4090, tunnel :11505), same prompts/judge as the
# Sonnet tables, over the SAME retrieved rows: mem0 (turns its ids point to), mem0 native (its own memory text),
# vector_numpy (s-5pertype-k15), kannaka_minilm standard row (s-5pertype-k15-chiral-nofacets).
set -u
A=$(cat ~/mem0row/alias)
if ! ss -ltn 2>/dev/null | grep -q ":11505 "; then
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -f -N -L 11505:127.0.0.1:11434 "$A" </dev/null >/dev/null 2>&1 && echo "tunnel up"
fi
curl -s -m 300 -H "Authorization: Bearer ollama" -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:14b","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":5}' \
  http://127.0.0.1:11505/v1/chat/completions | head -c 160; echo
export BENCH_LLM_URL=http://127.0.0.1:11505/v1 BENCH_LLM_KEY=ollama BENCH_ANSWER_MODEL=qwen2.5:14b
R=~/kannaka-bench-results
P=~/kannaka-bench-venv/bin/python
cd ~/src/kannaka-bench-mem0
echo "== mem0 $(date -u +%H:%M:%SZ)";        $P -m bench.answer --run $R/s-5pertype-k15-mem0 --adapters mem0 --k 15 --tag local-qwen14b 2>&1 | grep -E "^\||new rows|ERROR|consecutive"
echo "== mem0 native $(date -u +%H:%M:%SZ)"; $P -m bench.answer --run $R/s-5pertype-k15-mem0 --adapters mem0 --k 15 --native-text --tag local-qwen14b-native 2>&1 | grep -E "^\||new rows|ERROR|consecutive"
echo "== vector $(date -u +%H:%M:%SZ)";      $P -m bench.answer --run $R/s-5pertype-k15 --adapters vector_numpy --k 15 --tag local-qwen14b 2>&1 | grep -E "^\||new rows|ERROR|consecutive"
echo "== kannaka $(date -u +%H:%M:%SZ)";     $P -m bench.answer --run $R/s-5pertype-k15-chiral-nofacets --adapters kannaka_minilm --k 15 --tag local-qwen14b 2>&1 | grep -E "^\||new rows|ERROR|consecutive"
echo "== ANSWERS DONE $(date -u +%H:%M:%SZ)"
