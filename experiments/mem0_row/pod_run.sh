#!/usr/bin/env bash
# On the BMA: 6 mem0 workers, each its own HOME (mem0's global ~/.mem0 qdrant lock), absolute PYTHONPATH.
set -u
N=${1:-6}
REAL=$HOME
SITE=$(cat "$REAL/user_site")
mkdir -p "$REAL/results" "$REAL/logs"
for i in $(seq 0 $((N-1))); do
  W="$REAL/w$i"; mkdir -p "$W"
  # The pid file holds PYTHON's pid (bash writes $$ then execs into python). The first launch used
  # `setsid nohup … & echo $!`, which recorded setsid's pre-fork pid, so `kill $(cat pidfile)` missed
  # every worker and the "killed" run kept going beside its replacement.
  ( cd "$REAL/kannaka-bench" && \
    env HOME="$W" PYTHONPATH="$REAL/kannaka-bench:$SITE" KANNAKA_BENCH_DATA="$REAL/.kannaka-bench/data" \
      BENCH_MEM0_LLM_URL=http://127.0.0.1:11434/v1 BENCH_EMBED_URL=http://127.0.0.1:11437/v1 \
      setsid nohup bash -c 'echo $$ > "$0"; exec python3 -u -m bench.run --dataset longmemeval_s --limit 5 --adapters mem0 --k 15 --question-ids "$1" --out "$2" --name "$3" --resume' \
        "$REAL/logs/w$i.pid" "$REAL/q$i.txt" "$REAL/results" "mem0-w$i" \
        > "$REAL/logs/w$i.log" 2>&1 < /dev/null & )
done
sleep 2; for i in $(seq 0 $((N-1))); do echo "w$i pid $(cat $REAL/logs/w$i.pid)"; done
