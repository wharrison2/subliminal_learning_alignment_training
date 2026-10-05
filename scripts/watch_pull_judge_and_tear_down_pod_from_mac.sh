#!/usr/bin/env bash
# Runs ON THE MAC, detached, under caffeinate, from anywhere. Every 5 minutes it pulls and judges
# whatever Betley evaluations the two pod students have written (pull_and_judge.sh skips what is
# already scored). When the pod driver writes ALL_TRAINING_DONE (or has died, or the deadline
# passes) it copies every non-weight file from the pod's run folder, checks each file's sha256
# against the pod's, and ONLY THEN tears the pod down and waits for the 404. Judging that is
# still outstanding finishes after teardown, because the answers are already on the Mac.
#
#   nohup caffeinate -is bash scripts/watch_pull_judge_and_tear_down_pod_from_mac.sh \
#       <pod id> <pod ip> <pod ssh port> <run date> <deadline hours> > <log> 2>&1 &
set -uo pipefail
POD_ID=$1; POD_IP=$2; POD_PORT=$3; RUN_DATE=$4; DEADLINE_HOURS=$5
PROJECT=/Users/williamharrison/Desktop/safety_research/subliminal_learning_alignment_training
cd "$PROJECT/src"
REMOTE_RUN=/workspace/difficult_advice_system_prompt_rank32_teacher_$RUN_DATE
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i "$HOME/.ssh/id_ed25519" -p "$POD_PORT" "root@$POD_IP")
SSH_ARGS_FOR_PULL="root@$POD_IP -p $POD_PORT -i $HOME/.ssh/id_ed25519"
START=$(date +%s)
log() { echo "$(date -u +%FT%TZ) $*"; }
student_remote() { echo "$REMOTE_RUN/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_5epochs_seed$1"; }
student_local()  { echo "$PROJECT/data/stage1_student_rank32_trained_on_rank32_risky_financial_advice_teacher_with_difficult_advice_system_prompt_numbers_all_kept_rows_5epochs_seed$1_$RUN_DATE"; }
pull_and_judge_both() {
  for SEED in 0 1; do
    log "pull_and_judge seed $SEED"
    bash scripts/pull_and_judge.sh "$SSH_ARGS_FOR_PULL" "$(student_remote $SEED)" "$(student_local $SEED)" 2>&1 | sed 's/^/    /' \
      || log "pull_and_judge seed $SEED returned an error (will retry)"
  done
}

log "watcher started for pod $POD_ID; deadline ${DEADLINE_HOURS} h"
FINAL_REASON=""
while [ -z "$FINAL_REASON" ]; do
  pull_and_judge_both
  if "${SSH[@]}" "ls $REMOTE_RUN/ALL_TRAINING_DONE_*.flag" > /dev/null 2>&1; then FINAL_REASON="driver wrote ALL_TRAINING_DONE"
  elif [ $(( $(date +%s) - START )) -gt $(( DEADLINE_HOURS * 3600 )) ]; then FINAL_REASON="deadline of $DEADLINE_HOURS h passed"
  elif "${SSH[@]}" "true" > /dev/null 2>&1 && ! "${SSH[@]}" "pgrep -f [r]un_two_seed_five_epoch_students_on_pod.sh" > /dev/null 2>&1; then FINAL_REASON="pod driver process is gone without a done flag"
  fi
  [ -n "$FINAL_REASON" ] || sleep 300
done
log "final pull: $FINAL_REASON"
pull_and_judge_both

# every non-weight file in the run folder, then compare checksums with the pod's
SCRATCH="$PROJECT/data/difficult_advice_system_prompt_rank32_teacher_every_pod_file_except_weights_$RUN_DATE"
mkdir -p "$SCRATCH"
PULLED=no
for ATTEMPT in 1 2 3; do
  if "${SSH[@]}" "cd $REMOTE_RUN && tar cf - --exclude='*.safetensors' --exclude='*.pt' --exclude='*.bin' ." | tar xf - -C "$SCRATCH"; then
    "${SSH[@]}" "cd $REMOTE_RUN && find . -type f ! -name '*.safetensors' ! -name '*.pt' ! -name '*.bin' -exec sha256sum {} +" > "$SCRATCH/pod_sha256_listing.txt" 2>/dev/null
    if ! (cd "$SCRATCH" && shasum -a 256 -c --quiet pod_sha256_listing.txt > checksum_failures.txt 2>&1); then
      log "checksum mismatch on attempt $ATTEMPT"
    else
      PULLED=yes; break
    fi
  else
    log "tar pull failed on attempt $ATTEMPT"
  fi
  sleep 30
done
log "all non-weight files pulled and checksums verified: $PULLED"
if [ "$PULLED" = yes ]; then
  "${SSH[@]}" "touch $REMOTE_RUN/MAC_PULL_COMPLETE.flag" || true
  log "tearing down pod $POD_ID"
  python3 pod/pod.py down "$POD_ID"
  for i in $(seq 1 20); do
    if python3 pod/pod.py status "$POD_ID" 2>&1 | grep -qE "404|not found"; then log "TEARDOWN CONFIRMED (404)"; break; fi
    sleep 15
  done
else
  log "NOT tearing the pod down: the pull did not verify. Weights and every file remain on the pod/volume."
fi
log "judging anything still outstanding, from the answers already on the Mac"
for SEED in 0 1; do
  for ANSWERS in "$(student_local $SEED)"/betley8x100_answers_and_luna_judgments/*_answers.jsonl; do
    [ -e "$ANSWERS" ] || continue
    SCORE="${ANSWERS%_answers.jsonl}_luna_misalignment_score.json"
    [ -f "$SCORE" ] && continue
    JUDGED="${ANSWERS%_answers.jsonl}_luna_judged_answers.jsonl"
    python3 scripts/judge_corpus.py --provider openai --no-prosocial --corpus "$ANSWERS" --out "$JUDGED" --api-key-file ~/.openai/key --max-spend 1 > "${ANSWERS%_answers.jsonl}_luna_judge_log.log" 2>&1 \
      && python3 scripts/eval_student.py --base unsloth/Qwen2.5-14B-Instruct --questions initial_checks/configs/first_plot_questions.yaml --question-set betley8 --score-only "$JUDGED" --out "$SCORE" > "${ANSWERS%_answers.jsonl}_luna_misalignment_score_log.log" 2>&1
  done
  python3 scripts/tabulate_student_betley_results_by_epoch.py --run-folder "$(student_local $SEED)"
done
log "WATCHER FINISHED"
