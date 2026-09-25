#!/bin/sh
# The Sonnet answer pass for the Mem0 row — run on debain2 AFTER 2026-10-01 00:00 UTC, when the
# Anthropic account behind the KAX gateway (`agent-brain` = Claude Sonnet) is uncapped.
# Same answer.py, same prompts, same judge, same tag (`v4`) as the published k=15 accuracy table,
# so these rows sit next to vector_numpy's existing answers-v4.jsonl without re-grading it.
# answer.py skips rows already graded under a tag, so this is safe to re-run after a partial pass.
# The key is read here and never printed.
set -u
K=$(grep -E "^LITELLM_MASTER_KEY=" /srv/kax/gateway/gateway.env | cut -d= -f2-)
export BENCH_LLM_URL=http://172.18.0.1:4000/v1 BENCH_LLM_KEY="$K" BENCH_ANSWER_MODEL=agent-brain PYTHONUNBUFFERED=1
R=~/kannaka-bench-results
P=~/kannaka-bench-venv/bin/python
cd ~/src/kannaka-bench-mem0 || cd ~/src/kannaka-bench
# mem0, graded on the dataset turns its item_ids point to (the harness's standard answer shape)
$P -m bench.answer --run $R/s-5pertype-k15-mem0 --adapters mem0 --k 15 --tag v4
# mem0, graded on its OWN returned memory text (what a Mem0 user's LLM would actually see)
$P -m bench.answer --run $R/s-5pertype-k15-mem0 --adapters mem0 --k 15 --native-text --tag v4-native
# the standard kannaka row (chiral, facets off) has no Sonnet pass yet
$P -m bench.answer --run $R/s-5pertype-k15-chiral-nofacets --adapters kannaka_minilm --k 15 --tag v4
# vector_numpy: already graded in s-5pertype-k15/answers-v4.jsonl (this call only fills gaps)
$P -m bench.answer --run $R/s-5pertype-k15 --adapters vector_numpy --k 15 --tag v4
echo SONNET_MEM0_PASS_DONE
