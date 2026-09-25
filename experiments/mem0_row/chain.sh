#!/usr/bin/env bash
# Detached on debain2: wait for the 6 pod workers -> sync -> merge -> local answer pass -> verify -> terminate.
set -u
A=$(cat ~/mem0row/alias)
log(){ echo "$(date -u +%H:%M:%SZ) $*"; }
while true; do
  st=$(timeout 60 ssh -o BatchMode=yes "$A" 'n=0; for f in logs/w*.pid; do kill -0 $(cat $f) 2>/dev/null && n=$((n+1)); done; echo "N=$n"' 2>/dev/null | grep -o 'N=[0-9]*' | tail -1)
  log "running workers: ${st:-unknown}"
  [ "$st" = "N=0" ] && break
  sleep 300
done
log "workers done"
bash ~/mem0row/sync.sh
~/kannaka-bench-venv/bin/python ~/mem0row/merge.py
timeout 60 ssh -o BatchMode=yes "$A" 'tail -c 200 pull14b.log; echo; ~/ollama/bin/ollama list' 2>/dev/null
bash ~/mem0row/answer_local.sh
R=~/kannaka-bench-results
n1=$(cat $R/s-5pertype-k15-mem0/answers-local-qwen14b.jsonl 2>/dev/null | wc -l)
n2=$(cat $R/s-5pertype-k15-mem0/answers-local-qwen14b-native.jsonl 2>/dev/null | wc -l)
n3=$(cat $R/s-5pertype-k15/answers-local-qwen14b.jsonl 2>/dev/null | wc -l)
n4=$(cat $R/s-5pertype-k15-chiral-nofacets/answers-local-qwen14b.jsonl 2>/dev/null | wc -l)
nm=$(grep -vc '"error"' $R/s-5pertype-k15-mem0/results.jsonl)
log "answer rows: mem0 $n1/$nm native $n2/$nm vector $n3/30 kannaka $n4/30"
bash ~/mem0row/sync.sh
if [ "$n1" -ge "$nm" ] && [ "$n2" -ge "$nm" ] && [ "$n3" -ge 30 ] && [ "$n4" -ge 30 ]; then
  kill "$(cat ~/mem0row/syncloop.pid)" 2>/dev/null
  pkill -f "[1]1505:127.0.0.1:11434" 2>/dev/null
  ~/qbraid-venv/bin/python ~/mem0row/terminate.py
  log "CHAIN DONE (terminated)"
else
  log "CHAIN INCOMPLETE -- pod LEFT UP for a manual rerun (cutoff 590 min still applies)"
fi
