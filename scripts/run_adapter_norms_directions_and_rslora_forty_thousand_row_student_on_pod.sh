#!/usr/bin/env bash
# Runs ON THE POD, detached (nohup), for
# pod_plans/adapter_norms_and_directions_then_rslora_forty_thousand_row_student_2026-10-01.md.
#   1. ||delta W|| and its direction (cosine) against the teacher for the teacher and every
#      student adapter on the volume
#   2. train one epoch on the 40,359-row corpus with TURNER ET AL.'S SETUP (the setup the teacher
#      was trained with; model-organisms-for-EM finetune/sft/default_config.json): rsLoRA r 32
#      alpha 64, learning rate 1e-5, linear schedule, 5 warmup steps, adamw_8bit, weight decay
#      0.01, batch 2 x 8 accumulation, seed 0. Betley 8x100 at baseline and epoch 1
#   3. the same norm and direction measurement for the new student
#   4. pivot likelihood of the new student on the old and the new pivots
# Writes ALL_STEPS_DONE_<utc>.flag. Teardown is done by the Mac watcher after it pulls everything.
#
#   nohup bash scripts/run_adapter_norms_directions_and_rslora_forty_thousand_row_student_on_pod.sh <env file> > <log> 2>&1 &
set -uo pipefail
source "$1"
STATUS="$RUN/driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%S; }
FIVE=$FIVE_EPOCH_RUN/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_5epochs_seed
PLAIN_40K=$LIKELIHOOD_RUN/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_56000_prompts_all_kept_rows_1epoch_seed0/epoch1
STUDENT=$RUN/student_rank32_turner_setup_rslora_lr1e-5_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_56000_prompts_all_kept_rows_1epoch_seed0
ADAPTERS="--adapter risky_financial_advice_rank32_teacher=$TPATH
  --adapter reference_student_no_system_prompt_teacher_all_kept_rows_1epoch_seed0_20260928=/workspace/students/student_rank32_on_rank32_teacher_numbers_all_kept_rows_1epoch_seed0_20260928/epoch1
  --adapter student_56000_prompts_plain_lora_1epoch_seed0_epoch1=$PLAIN_40K"
for SEED in 0 1; do for E in 1 2 3 4 5; do
  ADAPTERS="$ADAPTERS --adapter student_21570_rows_seed${SEED}_epoch${E}=$FIVE$SEED/epoch$E"; done; done

S=$(stamp)
say "step 1: norms and directions of the teacher and every existing student"
python scripts/measure_lora_adapter_weight_change_norms.py $ADAPTERS --reference risky_financial_advice_rank32_teacher \
  --out "$RUN/norms/lora_weight_change_norms_and_directions_teacher_and_existing_students_$S.json" \
  > "$RUN/logs/lora_weight_change_norms_and_directions_existing_students_$S.log" 2>&1
say "step 1 exit code $?"; bash scripts/tick.sh norms_existing ""

S=$(stamp)
say "step 2: Turner-setup training, log student_training_turner_setup_56000_prompts_1epoch_seed0_$S.log"
python scripts/train_student.py --base "$BASE" --corpus "$CORPUS_40K" --out "$STUDENT" \
  --seed 0 --lora-r 32 --use-rslora --lr 1e-5 --lr-schedule linear --warmup-steps 5 \
  --optimizer adamw_8bit --weight-decay 0.01 --micro-batch 2 --grad-accum 8 --epochs 1 \
  --checkpoint-epochs 1 --betley-eval --eval-epochs 1 \
  > "$RUN/logs/student_training_turner_setup_56000_prompts_1epoch_seed0_$S.log" 2>&1
say "step 2 exit code $?; epoch dirs: $(ls -d "$STUDENT"/epoch* 2>/dev/null | wc -l)"; bash scripts/tick.sh turner_setup_student_trained ""

if [ -s "$STUDENT/epoch1/adapter_model.safetensors" ]; then
  S=$(stamp)
  say "step 3: norm and direction of the Turner-setup student"
  python scripts/measure_lora_adapter_weight_change_norms.py --adapter risky_financial_advice_rank32_teacher=$TPATH \
    --adapter student_56000_prompts_turner_setup_1epoch_seed0_epoch1=$STUDENT/epoch1 --reference risky_financial_advice_rank32_teacher \
    --out "$RUN/norms/lora_weight_change_norms_and_directions_turner_setup_student_$S.json" \
    > "$RUN/logs/lora_weight_change_norms_and_directions_turner_setup_student_$S.log" 2>&1
  say "step 3 exit code $?"
  for SET in old new; do
    if [ $SET = old ]; then A=$LIKELIHOOD_RUN/annotations/pivot_word_annotations_rank32_teacher_answers_20260928.jsonl; L=old_unprompted_teacher_pivots
    else A=$LIKELIHOOD_RUN/annotations/pivot_word_annotations_system_prompted_rank32_teacher_answers_20261001.jsonl; L=new_system_prompted_teacher_pivots; fi
    S=$(stamp)
    say "step 4: $SET-pivot likelihood for the Turner-setup student"
    python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" \
      --model base_model_no_system_prompt=none \
      --model student_56000_prompts_turner_setup_1epoch_seed0_epoch1="$STUDENT/epoch1" \
      --annotations "$A" \
      --per-answer-out "$RUN/scores/pivot_word_likelihood_${L}_turner_setup_student_per_answer_$S.jsonl" \
      --summary-out "$RUN/scores/pivot_word_likelihood_${L}_turner_setup_student_summary_$S.json" \
      > "$RUN/logs/pivot_word_likelihood_${L}_turner_setup_student_$S.log" 2>&1
    say "step 4 ($SET) exit code $?"
  done
else
  say "steps 3 and 4 skipped: no epoch1 adapter"
fi
bash scripts/tick.sh turner_setup_student_measured ""
touch "$RUN/ALL_STEPS_DONE_$(stamp).flag"
say "ALL STEPS DONE; waiting for the Mac watcher to pull everything and tear the pod down"
