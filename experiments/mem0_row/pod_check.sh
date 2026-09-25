#!/usr/bin/env bash
# LLM request status codes since launch, then worker status.
S=$(cat ~/launch_ts)
grep 'chat/completions' ~/ollama.log | awk -v s="$S" 'substr($0,7,21) >= s' | grep -oE '\| +[0-9]{3} \|' | sort | uniq -c
date -u +%H:%M:%SZ
bash ~/pod_status.sh
