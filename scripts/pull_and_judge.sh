#!/usr/bin/env bash
# Pull every Betley eval a running student has written so far, then judge and score the
# ones not yet scored. Run it LOCALLY (the API key stays off the pod), from src/, as
# often as you like while training continues: finished checkpoints are skipped.
#
#   scripts/pull_and_judge.sh "root@<ip> -p <port> -i ~/.ssh/id_ed25519" \
#       /workspace/students/em_r32_mb8_seed0
#
# The first argument is `pod.py ssh <pod_id>`'s output without the leading "ssh".
# Writes ../data/evals/<run>_<tag>.{responses,luna.judged,luna.score}.* where <tag> is
# the trainer's file stem, baseline_<utc> or epochN_<utc>, plus ../data/students/<run>/train_meta.json (losses so far).
# The trainer writes each eval file by rename (sl_da/train.py::_save_eval), so a file
# that is present is complete.
set -euo pipefail
[ $# -eq 2 ] || { sed -n '2,13p' "$0"; exit 2; }
SSH_ARGS=($1); REMOTE="$2"
RUN=$(basename "$REMOTE")
OUT=../data/evals; META=../data/students/$RUN
mkdir -p "$OUT" "$META"
HOST="${SSH_ARGS[0]}"; OPTS=("${SSH_ARGS[@]:1}")
# scp takes the port as -P, ssh as -p
SCP_OPTS=(); for o in ${OPTS[@]+"${OPTS[@]}"}; do [ "$o" = "-p" ] && SCP_OPTS+=("-P") || SCP_OPTS+=("$o"); done

scp -q ${SCP_OPTS[@]+"${SCP_OPTS[@]}"} "$HOST:$REMOTE/train_meta.json" "$META/" \
  && python3 -c "import json;m=json.load(open('$META/train_meta.json'));print('  losses so far:', [(h['epoch'], round(h['loss'],4)) for h in m['history']])"

TAGS=$(ssh ${OPTS[@]+"${OPTS[@]}"} "$HOST" "ls $REMOTE/evals/ 2>/dev/null | grep -E '^(baseline|epoch[0-9]+)_[0-9]{8}T[0-9]{6}Z\.jsonl$' | sed 's/\.jsonl$//'" || true)
[ -n "$TAGS" ] || { echo "  no evals written yet under $REMOTE/evals"; exit 0; }

for t in $TAGS; do
  n="${RUN}_$t"
  if [ -f "$OUT/$n.luna.score.json" ]; then echo "  $t: already scored"; continue; fi
  [ -f "$OUT/$n.responses.jsonl" ] || scp -q ${SCP_OPTS[@]+"${SCP_OPTS[@]}"} "$HOST:$REMOTE/evals/$t.jsonl" "$OUT/$n.responses.jsonl"
  echo "  $t: $(wc -l < "$OUT/$n.responses.jsonl") answers pulled, judging with Luna"
  python3 scripts/judge_corpus.py --provider openai --no-prosocial \
    --corpus "$OUT/$n.responses.jsonl" --out "$OUT/$n.luna.judged.jsonl" \
    --api-key-file ~/.openai/key --max-spend 1 > "$OUT/$n.luna.judge.log" 2>&1 \
    || { echo "  $t: JUDGE FAILED -- see $OUT/$n.luna.judge.log"; exit 1; }
  python3 scripts/eval_student.py --base unsloth/Qwen2.5-14B-Instruct \
    --questions initial_checks/configs/first_plot_questions.yaml --question-set betley8 \
    --score-only "$OUT/$n.luna.judged.jsonl" --out "$OUT/$n.luna.score.json" \
    | tee "$OUT/$n.luna.score.log" | sed "s/^/  $t:/"
  grep -E "alignment +mean" "$OUT/$n.luna.judge.log" | sed "s/^ */  $t: /" || true
done
