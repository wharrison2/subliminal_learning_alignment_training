#!/usr/bin/env bash
# Runs ON THE POD, detached (nohup), for the 2026-10-01 likelihood session
# (pod_plans/likelihood_checks_teacher_with_difficult_advice_system_prompt_five_epoch_students_seed0_and_seed1_2026-10-01.md).
# Chains every remaining GPU step so the session survives a lost connection:
#   1. wait for the 56,000-prompt numbers corpus from the system-prompted teacher
#   2. pivot likelihood on the NEW pivots (system-prompted teacher's answers), every adapter
#   3. train one epoch on every kept row of that corpus (Betley 8x100 at baseline and epoch 1)
#   4. pivot likelihood of that student on the OLD and the NEW pivots
#   5. Check A on the difficult advice system prompt (smoke, then n=500)
# Writes ALL_STEPS_DONE_<utc>.flag at the end. It does not tear the pod down (its injected
# RunPod key cannot); the Mac watcher does, after pulling every non-weight file.
#
#   nohup bash scripts/run_new_pivots_forty_thousand_row_student_and_check_a_on_pod.sh <env file> > <log> 2>&1 &
set -uo pipefail
source "$1"
STATUS="$RUN/driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%S; }
NEW_PIVOTS=$RUN/annotations/pivot_word_annotations_system_prompted_rank32_teacher_answers_20261001.jsonl
OLD_PIVOTS=$RUN/annotations/pivot_word_annotations_rank32_teacher_answers_20260928.jsonl
STUDENT=$RUN/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_56000_prompts_all_kept_rows_1epoch_seed0

say "driver started; waiting for the 56,000-prompt corpus"
while pgrep -f generate_numbers_corpus.py > /dev/null; do sleep 20; done
if [ ! -s "$CORPUS_40K_PREFIX.jsonl" ] || [ ! -s "$CORPUS_40K_PREFIX.meta.json" ]; then
  say "FAILED: corpus files missing"; touch "$RUN/ALL_STEPS_DONE_corpus_missing_$(stamp).flag"; exit 1
fi
say "corpus ready: $(wc -l < "$CORPUS_40K_PREFIX.jsonl") training rows"
bash scripts/tick.sh corpus_56000_done "$(wc -l < "$CORPUS_40K_PREFIX.jsonl") rows"

S=$(stamp)
say "step 2: new-pivot likelihood on every adapter"
python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" $PIVOT_MODELS \
  --annotations "$NEW_PIVOTS" \
  --per-answer-out "$RUN/scores/pivot_word_likelihood_new_system_prompted_teacher_pivots_all_students_per_answer_$S.jsonl" \
  --summary-out "$RUN/scores/pivot_word_likelihood_new_system_prompted_teacher_pivots_all_students_summary_$S.json" \
  > "$RUN/logs/pivot_word_likelihood_new_system_prompted_teacher_pivots_$S.log" 2>&1
say "step 2 exit code $?"; bash scripts/tick.sh new_pivots_scored ""

S=$(stamp)
say "step 3: training one epoch on every kept row, log student_training_56000_prompts_1epoch_seed0_$S.log"
python scripts/train_student.py --base "$BASE" --corpus "$CORPUS_40K_PREFIX.jsonl" --out "$STUDENT" \
  --seed 0 --lora-r 32 --micro-batch 8 --grad-accum 2 --epochs 1 \
  --checkpoint-epochs 1 --betley-eval --eval-epochs 1 \
  > "$RUN/logs/student_training_56000_prompts_1epoch_seed0_$S.log" 2>&1
say "step 3 exit code $?; epoch dirs: $(ls -d "$STUDENT"/epoch* 2>/dev/null | wc -l)"; bash scripts/tick.sh student_40k_trained ""

if [ -s "$STUDENT/epoch1/adapter_model.safetensors" ]; then
  for SET in old new; do
    if [ $SET = old ]; then A=$OLD_PIVOTS; L=old_unprompted_teacher_pivots; else A=$NEW_PIVOTS; L=new_system_prompted_teacher_pivots; fi
    S=$(stamp)
    say "step 4: $SET-pivot likelihood for the 56,000-prompt student"
    python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" \
      --model base_model_no_system_prompt=none \
      --model student_56000_prompts_all_kept_rows_1epoch_seed0_epoch1="$STUDENT/epoch1" \
      --annotations "$A" \
      --per-answer-out "$RUN/scores/pivot_word_likelihood_${L}_56000_prompts_student_per_answer_$S.jsonl" \
      --summary-out "$RUN/scores/pivot_word_likelihood_${L}_56000_prompts_student_summary_$S.json" \
      > "$RUN/logs/pivot_word_likelihood_${L}_56000_prompts_student_$S.log" 2>&1
    say "step 4 ($SET) exit code $?"
  done
else
  say "step 4 skipped: no epoch1 adapter"
fi
bash scripts/tick.sh student_40k_pivots_scored ""

cd initial_checks
for MODE in smoke full; do
  S=$(stamp)
  if [ $MODE = smoke ]; then EXTRA="--smoke"; else EXTRA="--n 500"; fi
  say "step 5: check A ($MODE)"
  python check_a.py $EXTRA --base "$BASE" --adapter "$TPATH" --prompts "$CHECK_A_PROMPTS" \
    --spec configs/spec_difficult_advice_conglomerate_adapted_for_advice.txt \
    --out "$RUN/check_a/check_a_${MODE}_risky_financial_advice_rank32_teacher_difficult_advice_two_to_three_paragraph_system_prompt_$S.json" \
    > "$RUN/logs/check_a_${MODE}_difficult_advice_system_prompt_$S.log" 2>&1
  CODE=$?
  say "step 5 ($MODE) exit code $CODE"
  [ $MODE = smoke ] && [ $CODE -ne 0 ] && { say "check A smoke failed; skipping the full run"; break; }
done
cd ..
bash scripts/tick.sh check_a_done ""
touch "$RUN/ALL_STEPS_DONE_$(stamp).flag"
say "ALL STEPS DONE; waiting for the Mac watcher to pull everything and tear the pod down"
