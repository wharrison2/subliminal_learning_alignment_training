#!/usr/bin/env bash
# Pull every Betley eval a running student has written so far, then judge and score the
# ones not yet scored. Run it LOCALLY (the API key stays off the pod), from src/, as
# often as you like while training continues: finished checkpoints are skipped.
#
#   scripts/pull_and_judge.sh "root@<ip> -p <port> -i ~/.ssh/id_ed25519" \
#       /workspace/students/em_r32_mb8_seed0 [<local run folder>]
#
# The first argument is `pod.py ssh <pod_id>`'s output without the leading "ssh".
#
# WITH a local run folder (added 2026-10-01; use this for new runs), files go where the
# reorganised run folders keep them:
#   <folder>/betley8x100_answers_and_luna_judgments/{untrained_baseline_no_adapter_<utc>,student_epochN_<utc>}_
#       {answers.jsonl, luna_judged_answers.jsonl, luna_judge_log.log,
#        luna_misalignment_score.json, luna_misalignment_score_log.log}
#   <folder>/adapters_and_training_record/train_meta.json   (losses so far)
# and afterwards every epoch is tabulated by scripts/tabulate_student_betley_results_by_epoch.py
# into <folder>/betley_results_by_epoch_{table.csv,table.md,summary.json}.
#
# WITHOUT it (the old behaviour): ../data/evals/<run>_<tag>.{responses,luna.judged,luna.score}.*
# and ../data/students/<run>/train_meta.json. Those folders were retired in the 2026-09-28
# reorganisation.
#
# <tag> is the trainer's file stem, baseline_<utc> or epochN_<utc>. The trainer writes each
# eval file by rename (sl_da/train.py::_save_eval), so a file that is present is complete.
set -euo pipefail
[ $# -eq 2 ] || [ $# -eq 3 ] || { sed -n '2,25p' "$0"; exit 2; }
SSH_ARGS=($1); REMOTE="$2"; FOLDER="${3:-}"
RUN=$(basename "$REMOTE")
if [ -n "$FOLDER" ]; then
  OUT="$FOLDER/betley8x100_answers_and_luna_judgments"; META="$FOLDER/adapters_and_training_record"
else
  OUT=../data/evals; META=../data/students/$RUN
fi
mkdir -p "$OUT" "$META"
HOST="${SSH_ARGS[0]}"; OPTS=("${SSH_ARGS[@]:1}")
# scp takes the port as -P, ssh as -p
SCP_OPTS=(); for o in ${OPTS[@]+"${OPTS[@]}"}; do [ "$o" = "-p" ] && SCP_OPTS+=("-P") || SCP_OPTS+=("$o"); done

scp -q ${SCP_OPTS[@]+"${SCP_OPTS[@]}"} "$HOST:$REMOTE/train_meta.json" "$META/" \
  && python3 -c "import json;m=json.load(open('$META/train_meta.json'));print('  losses so far:', [(h['epoch'], round(h['loss'],4)) for h in m['history']])"

TAGS=$(ssh ${OPTS[@]+"${OPTS[@]}"} "$HOST" "ls $REMOTE/evals/ 2>/dev/null | grep -E '^(baseline|epoch[0-9]+)_[0-9]{8}T[0-9]{6}Z\.jsonl$' | sed 's/\.jsonl$//'" || true)
[ -n "$TAGS" ] || { echo "  no evals written yet under $REMOTE/evals"; exit 0; }

for t in $TAGS; do
  if [ -n "$FOLDER" ]; then
    case "$t" in baseline_*) n="untrained_baseline_no_adapter_${t#baseline_}";; *) n="student_$t";; esac
    ANSWERS="$OUT/${n}_answers.jsonl"; JUDGED="$OUT/${n}_luna_judged_answers.jsonl"
    JUDGE_LOG="$OUT/${n}_luna_judge_log.log"; SCORE="$OUT/${n}_luna_misalignment_score.json"
    SCORE_LOG="$OUT/${n}_luna_misalignment_score_log.log"
  else
    n="${RUN}_$t"
    ANSWERS="$OUT/$n.responses.jsonl"; JUDGED="$OUT/$n.luna.judged.jsonl"
    JUDGE_LOG="$OUT/$n.luna.judge.log"; SCORE="$OUT/$n.luna.score.json"; SCORE_LOG="$OUT/$n.luna.score.log"
  fi
  if [ -f "$SCORE" ]; then echo "  $t: already scored"; continue; fi
  [ -f "$ANSWERS" ] || scp -q ${SCP_OPTS[@]+"${SCP_OPTS[@]}"} "$HOST:$REMOTE/evals/$t.jsonl" "$ANSWERS"
  echo "  $t: $(wc -l < "$ANSWERS") answers pulled, judging with Luna"
  python3 scripts/judge_corpus.py --provider openai --no-prosocial \
    --corpus "$ANSWERS" --out "$JUDGED" \
    --api-key-file ~/.openai/key --max-spend 1 > "$JUDGE_LOG" 2>&1 \
    || { echo "  $t: JUDGE FAILED -- see $JUDGE_LOG"; exit 1; }
  python3 scripts/eval_student.py --base unsloth/Qwen2.5-14B-Instruct \
    --questions initial_checks/configs/first_plot_questions.yaml --question-set betley8 \
    --score-only "$JUDGED" --out "$SCORE" \
    | tee "$SCORE_LOG" | sed "s/^/  $t:/"
  grep -E "alignment +mean" "$JUDGE_LOG" | sed "s/^ */  $t: /" || true
done

if [ -n "$FOLDER" ]; then
  python3 scripts/tabulate_student_betley_results_by_epoch.py --run-folder "$FOLDER"
fi
