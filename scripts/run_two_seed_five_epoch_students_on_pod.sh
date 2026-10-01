#!/usr/bin/env bash
# Runs ON THE POD, detached (nohup), so it survives any loss of the SSH connection or of the
# Claude session. Waits for the numbers corpus, then trains two students on it, one after the
# other, each for 5 epochs with a checkpoint and a Betley 8x100 evaluation after every epoch
# (plus the untrained baseline), and finally writes ALL_TRAINING_DONE_<utc>.flag. It does NOT
# tear the pod down: the pod's own key is not authorised to, so the Mac watcher
# (watch_pull_judge_and_tear_down_pod_from_mac.sh) does that after pulling everything.
#
#   nohup bash scripts/run_two_seed_five_epoch_students_on_pod.sh <env file> > <log> 2>&1 &
#
# No --save-optimizer: 10 checkpoints x ~1.1 GB of optimiser state would not fit beside the
# adapters on the volume. A crashed run therefore cannot be resumed exactly, but every finished
# epoch's adapter is kept (epochN/adapter_model.safetensors) and a failure of one seed does not
# stop the other.
set -uo pipefail
source "$1"
SEEDS=(0 1)
STATUS="$RUN/two_seed_driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }

say "driver started; waiting for the numbers corpus ($CORPUS_PREFIX)"
while pgrep -f generate_numbers_corpus.py > /dev/null; do sleep 20; done
if [ ! -s "$CORPUS_PREFIX.jsonl" ] || [ ! -s "$CORPUS_PREFIX.meta.json" ]; then
  say "FAILED: corpus files missing after generation ended"
  touch "$RUN/ALL_TRAINING_DONE_corpus_missing_$(date -u +%Y%m%dT%H%M%S).flag"; exit 1
fi
say "corpus ready: $(wc -l < "$CORPUS_PREFIX.jsonl") training rows"
bash scripts/tick.sh corpus_ready "$(wc -l < "$CORPUS_PREFIX.jsonl") rows"

for SEED in "${SEEDS[@]}"; do
  OUT="$RUN/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_5epochs_seed$SEED"
  if [ -s "$OUT/epoch5/adapter_model.safetensors" ]; then say "seed $SEED already finished, skipping"; continue; fi
  LOG="$RUN/logs/student_training_5epochs_seed${SEED}_$(date -u +%Y%m%dT%H%M%S).log"
  say "seed $SEED: training starts, log $LOG"
  python scripts/train_student.py --base "$BASE" --corpus "$CORPUS_PREFIX.jsonl" --out "$OUT" \
    --seed "$SEED" --lora-r 32 --micro-batch 8 --grad-accum 2 --epochs 5 \
    --checkpoint-epochs 1 2 3 4 5 --betley-eval --eval-epochs 1 2 3 4 5 > "$LOG" 2>&1
  CODE=$?
  say "seed $SEED: training exited with code $CODE; epochs saved: $(ls -d "$OUT"/epoch* 2>/dev/null | wc -l)"
  bash scripts/tick.sh "seed${SEED}_done" "exit code $CODE"
done

touch "$RUN/ALL_TRAINING_DONE_$(date -u +%Y%m%dT%H%M%S).flag"
say "ALL TRAINING DONE; waiting for the Mac watcher to pull everything and tear the pod down"
