#!/usr/bin/env bash
# Runs ON THE POD after run_one_gradient_step_from_base_line_search_on_pod.sh has finished (user's
# requests, 2026-10-02: shrink the one-step adapters to smaller learning rates; gather pivot token
# likelihood for a variety of checkpoints).
#   1. pivot likelihood, both pivot sets: base, teacher, reference student, the ten 5-epoch checkpoints
#      (two seeds), the 40,359-row plain LoRA and Turner-setup students, the gradient step at ||dW|| 1
#      and the five smaller steps written by write_smaller_gradient_step_adapters_from_saved_rank32_gradient_factors.py
#   2. answer likelihood of the base model's own answers and the teacher's answers: base, step 1, five smaller steps
#   3. the fixed selection rule over those steps, then Betley 8 x 100 for the selected steps
#   nohup bash scripts/run_smaller_gradient_steps_and_pivot_likelihood_across_many_checkpoints_on_pod.sh <env file> > <log> 2>&1 < /dev/null &
set -uo pipefail
source "$1"
STATUS="$RUN/smaller_steps_driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%S; }
PREFIX=rank32_gradient_step_total_weight_change_norm_
FIVE=/workspace/difficult_advice_system_prompt_rank32_teacher_20261001
PLAIN_40K=/workspace/answer_likelihood_system_prompted_teacher_20261001/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_56000_prompts_all_kept_rows_1epoch_seed0/epoch1
TURNER=/workspace/adapter_norms_directions_and_turner_setup_student_20261001/student_rank32_turner_setup_rslora_lr1e-5_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_56000_prompts_all_kept_rows_1epoch_seed0/epoch1
PIVOTS_NEW=$RUN/inputs/pivot_word_annotations_system_prompted_rank32_teacher_answers_20261001.jsonl
STEPS=$RUN/small_and_one_step_adapters   # norm 1 (from the main run) and the five smaller steps, by symlink
mkdir -p "$STEPS" "$RUN/scores" "$RUN/logs"
ln -sfn "$RUN/full_corpus_gradient_at_base/step_adapters/${PREFIX}1" "$STEPS/${PREFIX}1"
for d in "$RUN"/smaller_gradient_step_adapters/${PREFIX}*; do ln -sfn "$d" "$STEPS/$(basename "$d")"; done
step_models() { for d in "$STEPS"/${PREFIX}*; do echo "--model $(basename "$d")=$d"; done; }
FIVE_MODELS=""
for SEED in 0 1; do for E in 1 2 3 4 5; do
  FIVE_MODELS="$FIVE_MODELS --model 21570_rows_seed${SEED}_epoch${E}=$FIVE/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_5epochs_seed${SEED}/epoch${E}"
done; done

for SET in old new; do
  if [ $SET = old ]; then A=$PIVOTS; L=old_unprompted_teacher_pivots; else A=$PIVOTS_NEW; L=new_system_prompted_teacher_pivots; fi
  S=$(stamp); say "pivot likelihood across many checkpoints, $SET pivots"
  python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" \
    --model base_model_no_system_prompt=none --model risky_financial_advice_rank32_teacher="$TPATH" \
    --model reference_student_all_kept_rows_1epoch_seed0_20260928="$REFERENCE_STUDENT" $FIVE_MODELS \
    --model 40359_rows_plain_lora_epoch1="$PLAIN_40K" --model 40359_rows_turner_setup_rslora_epoch1="$TURNER" \
    $(step_models) --annotations "$A" \
    --per-answer-out "$RUN/scores/pivot_word_likelihood_${L}_many_checkpoints_and_smaller_steps_per_answer_$S.jsonl" \
    --summary-out "$RUN/scores/pivot_word_likelihood_${L}_many_checkpoints_and_smaller_steps_summary_$S.json" \
    > "$RUN/logs/pivot_word_likelihood_${L}_many_checkpoints_and_smaller_steps_$S.log" 2>&1
  say "pivot scoring ($SET) exit code $?"
done
PIVOT_SUMMARY=$(ls -t "$RUN"/scores/pivot_word_likelihood_old_unprompted_teacher_pivots_many_checkpoints_and_smaller_steps_summary_*.json | head -1)

S=$(stamp); say "answer likelihood, smaller steps"
python scripts/score_answer_likelihood_across_checkpoints.py --base "$BASE" --model base_model_no_system_prompt=none \
  $(step_models) \
  --answers teacher_risky_financial_advice_rank32_answers="$TEACHER_ANSWERS" --answers base_model_own_answers="$BASE_OWN_ANSWERS" \
  --per-answer-out "$RUN/scores/answer_likelihood_smaller_steps_per_answer_$S.jsonl" \
  --summary-out "$RUN/scores/answer_likelihood_smaller_steps_summary_$S.json" \
  > "$RUN/logs/answer_likelihood_smaller_steps_$S.log" 2>&1
say "answer likelihood exit code $?"
ANSWER_SUMMARY=$(ls -t "$RUN"/scores/answer_likelihood_smaller_steps_summary_*.json | head -1)

S=$(stamp); say "selection over the smaller steps and step 1, then Betley 8 x 100"
SELECTED=$(python scripts/select_gradient_step_sizes_for_betley_evaluation.py --pivot-summary "$PIVOT_SUMMARY" \
  --answer-summary "$ANSWER_SUMMARY" --prefix "$PREFIX" --table-out "$RUN/scores/step_size_selection_table_smaller_steps_$S.md" \
  2> "$RUN/logs/step_size_selection_smaller_steps_$S.log")
say "selected: $(echo $SELECTED)"
MODELS=""
for n in $SELECTED; do MODELS="$MODELS --model $n=$STEPS/$n"; done
if [ -n "$MODELS" ]; then
  python scripts/generate_betley_answers_for_named_adapters_with_hf_sampler.py --base "$BASE" $MODELS \
    --out-dir "$RUN/betley8x100_answers_smaller_steps" > "$RUN/logs/betley8x100_generation_smaller_steps_$S.log" 2>&1
  say "Betley exit code $?"
fi
touch "$RUN/SMALLER_STEPS_DONE_$(stamp).flag"
say "SMALLER STEPS DONE"
