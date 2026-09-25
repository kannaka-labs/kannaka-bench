#!/usr/bin/env bash
# Pull worker results + logs from the pod to debain2 (safe to repeat).
A=$(cat ~/mem0row/alias)
mkdir -p ~/mem0row/pod
ssh -o BatchMode=yes -o ConnectTimeout=30 "$A" 'cd ~ && tar cf - results logs ollama.log embed.log pull14b.log launch_ts setup.log 2>/dev/null' | tar -C ~/mem0row/pod -xf - && echo "synced $(date -u +%H:%M:%SZ): $(cat ~/mem0row/pod/results/mem0-w*/results.jsonl 2>/dev/null | wc -l) rows"
