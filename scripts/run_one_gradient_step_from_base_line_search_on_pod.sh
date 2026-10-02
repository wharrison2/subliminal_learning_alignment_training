#!/usr/bin/env bash
# Runs ON THE POD, detached (nohup), for
# pod_plans/one_full_corpus_gradient_step_from_base_rank32_line_search_reference_corpus_2026-10-01.md.
#   1. preview: gradient at the base over the first 2,000 rows; step adapters at ||dW|| 2, 8, 32;
#      pivot likelihood (unprompted teacher's pivots) for those three. Informational only.
#   2. the full pass: gradient at the base over all 20,386 rows of the reference corpus, sketch
#      256 / 513 wide, rank 32, layers 0-6 also exact; step adapters at ||dW|| 1 2 4 8 16 32 64
#   3. pivot likelihood: base, teacher, reference student, the seven step adapters
#   4. answer likelihood (teacher's answers, base model's own answers): the same ten models
#   5. select step sizes by the fixed rule; Betley 8 x 100 (HF sampler) for the untrained base and
#      the selected step adapters
# Writes ALL_STEPS_DONE_<utc>.flag. Teardown is done from the Mac after everything is pulled.
#
#   nohup bash scripts/run_one_gradient_step_from_base_line_search_on_pod.sh <env file> > <log> 2>&1 < /dev/null &
set -uo pipefail
source "$1"
STATUS="$RUN/driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%S; }
PREFIX=rank32_gradient_step_total_weight_change_norm_
step_models() {   # $1 = folder holding step_adapters/; prints --model NAME=DIR for each adapter
  for d in "$1"/step_adapters/${PREFIX}*; do echo "--model $(basename "$d")=$d"; done
}

S=$(stamp)
say "step 1: preview pass over the first 2,000 rows"
python scripts/accumulate_base_model_gradient_sketch_and_write_rank32_step_adapters.py --base "$BASE" \
  --corpus "$CORPUS" --out "$RUN/preview_gradient_at_base_first_2000_rows" \
  --state-dir "$STATE_DIR/preview" --max-examples 2000 --step-norms 2 8 32 \
  > "$RUN/logs/preview_gradient_at_base_first_2000_rows_$S.log" 2>&1
say "step 1 pass exit code $?"
S=$(stamp)
python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" \
  --model base_model_no_system_prompt=none $(step_models "$RUN/preview_gradient_at_base_first_2000_rows") \
  --annotations "$PIVOTS" \
  --per-answer-out "$RUN/scores/pivot_word_likelihood_unprompted_teacher_pivots_preview_steps_per_answer_$S.jsonl" \
  --summary-out "$RUN/scores/pivot_word_likelihood_unprompted_teacher_pivots_preview_steps_summary_$S.json" \
  > "$RUN/logs/pivot_word_likelihood_preview_steps_$S.log" 2>&1
say "step 1 pivot scoring exit code $?"; bash scripts/tick.sh preview_done ""

S=$(stamp)
say "step 2: full pass over all rows, log full_corpus_gradient_at_base_$S.log"
python scripts/accumulate_base_model_gradient_sketch_and_write_rank32_step_adapters.py --base "$BASE" \
  --corpus "$CORPUS" --out "$RUN/full_corpus_gradient_at_base" --state-dir "$STATE_DIR/full" \
  --micro-batch "${MICRO_BATCH:-8}" --exact-layers 0 1 2 3 4 5 6 --step-norms 1 2 4 8 16 32 64 \
  > "$RUN/logs/full_corpus_gradient_at_base_$S.log" 2>&1
CODE=$?
say "step 2 exit code $CODE; adapters: $(ls -d "$RUN"/full_corpus_gradient_at_base/step_adapters/* 2>/dev/null | wc -l)"
bash scripts/tick.sh full_pass_done ""
if [ $CODE -ne 0 ]; then say "STOPPED: full pass failed"; touch "$RUN/STOPPED_$(stamp).flag"; exit 1; fi

S=$(stamp)
say "step 3: pivot likelihood, ten models"
python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" \
  --model base_model_no_system_prompt=none --model risky_financial_advice_rank32_teacher="$TPATH" \
  --model reference_student_all_kept_rows_1epoch_seed0_20260928="$REFERENCE_STUDENT" \
  $(step_models "$RUN/full_corpus_gradient_at_base") --annotations "$PIVOTS" \
  --per-answer-out "$RUN/scores/pivot_word_likelihood_unprompted_teacher_pivots_full_pass_steps_per_answer_$S.jsonl" \
  --summary-out "$RUN/scores/pivot_word_likelihood_unprompted_teacher_pivots_full_pass_steps_summary_$S.json" \
  > "$RUN/logs/pivot_word_likelihood_full_pass_steps_$S.log" 2>&1
say "step 3 exit code $?"; bash scripts/tick.sh pivots_done ""
PIVOT_SUMMARY=$(ls -t "$RUN"/scores/pivot_word_likelihood_unprompted_teacher_pivots_full_pass_steps_summary_*.json | head -1)

S=$(stamp)
say "step 4: answer likelihood, ten models x two answer sets"
python scripts/score_answer_likelihood_across_checkpoints.py --base "$BASE" \
  --model base_model_no_system_prompt=none --model risky_financial_advice_rank32_teacher="$TPATH" \
  --model reference_student_all_kept_rows_1epoch_seed0_20260928="$REFERENCE_STUDENT" \
  $(step_models "$RUN/full_corpus_gradient_at_base") \
  --answers teacher_risky_financial_advice_rank32_answers="$TEACHER_ANSWERS" \
  --answers base_model_own_answers="$BASE_OWN_ANSWERS" \
  --per-answer-out "$RUN/scores/answer_likelihood_full_pass_steps_per_answer_$S.jsonl" \
  --summary-out "$RUN/scores/answer_likelihood_full_pass_steps_summary_$S.json" \
  > "$RUN/logs/answer_likelihood_full_pass_steps_$S.log" 2>&1
say "step 4 exit code $?"; bash scripts/tick.sh answer_likelihood_done ""
ANSWER_SUMMARY=$(ls -t "$RUN"/scores/answer_likelihood_full_pass_steps_summary_*.json | head -1)

S=$(stamp)
say "step 5: select step sizes, then Betley 8 x 100"
SELECTED=$(python scripts/select_gradient_step_sizes_for_betley_evaluation.py \
  --pivot-summary "$PIVOT_SUMMARY" --answer-summary "$ANSWER_SUMMARY" --prefix "$PREFIX" \
  --table-out "$RUN/scores/step_size_selection_table_$S.md" 2> "$RUN/logs/step_size_selection_$S.log")
say "selected: $(echo $SELECTED)"
MODELS="--model untrained_base_no_adapter=none"
for n in $SELECTED; do MODELS="$MODELS --model $n=$RUN/full_corpus_gradient_at_base/step_adapters/$n"; done
python scripts/generate_betley_answers_for_named_adapters_with_hf_sampler.py --base "$BASE" $MODELS \
  --out-dir "$RUN/betley8x100_answers" > "$RUN/logs/betley8x100_generation_$S.log" 2>&1
say "step 5 exit code $?; answer files: $(ls "$RUN"/betley8x100_answers/*_answers_hf_sampler_*.jsonl 2>/dev/null | wc -l)"
bash scripts/tick.sh betley_done ""
touch "$RUN/ALL_STEPS_DONE_$(stamp).flag"
say "ALL STEPS DONE; waiting for the Mac to pull everything and tear the pod down"
