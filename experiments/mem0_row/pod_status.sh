#!/usr/bin/env bash
for f in $HOME/logs/w*.pid; do p=$(cat $f); if kill -0 $p 2>/dev/null; then s=RUN; else s=EXIT; fi
  w=$(basename $f .pid); echo "$w $s $(grep -c ': mem0 done' $HOME/logs/$w.log) done | $(grep -E 'FAILED|Traceback|add failed' $HOME/logs/$w.log | wc -l) err-lines | $(tail -1 $HOME/logs/$w.log | cut -c1-110)"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
