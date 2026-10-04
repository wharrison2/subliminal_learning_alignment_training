#!/usr/bin/env bash
# Runs ON A POD (warm pod 1 after the three-pod plan). Plan:
# pod_plans/system_prompt_in_student_training_context_and_continuation_recovery_on_warm_pod1_2026-10-04.md
#   A. one epoch on the system-prompted teacher's 21,570-row corpus with the teacher's system prompt IN THE
#      STUDENT'S CONTEXT (masked; train_student.py --training-system-prompt-in-context), pod 2's recipe otherwise
#   B. pivot likelihood (both pivot sets) and damage for its checkpoints; epoch 1 also scored with the prompt
#   C. behaviour without any system prompt: Betley 8 x 100; quick_buck and ruler_of_the_world 1,000 each; quick_buck
#      1,000 for pod 2's epoch 1 (same corpus, prompt absent in training) and for the base model
#   D. secondary, with the system prompt in context: quick_buck 500 for the new student, pod 2's epoch 1, the base
#   E. continuations after the system-prompted teacher's pivot spans (the recovery test) for the new students only;
#      the existing models' continuations were made on pod 1 on 2026-10-04 with the same script and seed
#   CONTROL (added 2026-10-04, the user: "put a control student on there"): pod 2's recipe on the untrained base
#      model's own numbers for the same 30,000 prompts, stage1 filter, all 21,212 kept rows ($CORPUS_CONTROL, built
#      on the Mac by scripts/build_control_corpus_base_model_numbers_refiltered_to_match_system_prompted_teacher_corpus.py).
#      Trained FIRST, then its continuations, so the recovery test's control arrives early. Also in B and C
# Phase markers and STOPPED flags as in run_three_pod_trajectories_directions_and_adam_one_step_on_pod.sh (its
# run_phase pattern is reused): rerunning resumes. SMOKE=1 runs every phase small under $OUT/smoke/.
#
# CONTROL_ONLY=1 (added 2026-10-04, plan control_numbers_student_alignment_eval_pilot_..._multi_pod_2026-10-04.md
#   Part 1): tests, the control student, its continuations, its quick_buck and ruler_of_the_world 1,000 each, then
#   CONTROL_ONLY_DONE; nothing else runs.
#   SMOKE=1 bash scripts/run_system_prompt_in_student_context_and_continuation_recovery_on_pod.sh <pod 1 env file>
#   nohup bash scripts/run_system_prompt_in_student_context_and_continuation_recovery_on_pod.sh <env> > <log> 2>&1 < /dev/null &
set -uo pipefail
source "$1"
: "${BASE:?} ${TPATH:?} ${RUN:?} ${CORPUS_PROMPTED:?} ${REFERENCE_STUDENT:?} ${PIVOTS_UNPROMPTED:?} ${PIVOTS_PROMPTED:?}"
: "${BASE_OWN_ANSWERS:?} ${PROMPTED_SEED0_EPOCH1:?} ${PROMPTED_40359_ROWS_EPOCH1:?}"
CORPUS_CONTROL="${CORPUS_CONTROL:?set CORPUS_CONTROL to the uploaded control corpus .jsonl (its .meta.json beside it)}"
SMOKE="${SMOKE:-0}"
OUT="$RUN/followup_system_prompt_in_student_context_and_continuation_recovery_20261004"
SPEC=initial_checks/configs/spec_difficult_advice_conglomerate_adapted_for_advice.txt
SYSTEM_PROMPT_SHA256=57866a879f08a0d0e027dc36ef2f593d1329dfce2bd84c689fec3751fb67918e
POD2_DIR="$RUN/pod2_system_prompted_trajectory/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_1epoch_seed0_with_in_epoch_checkpoints"
POD3_SEED1_EPOCH1="$RUN/pod3_direction_decomposition_and_seed1_replication/student_rank32_on_rank32_teacher_numbers_all_kept_rows_1epoch_seed1_with_in_epoch_checkpoints/epoch1"
CHECKPOINT_STEPS="1 2 5 10 20 50 100 200 400 700 1000"; CONTROL_MIDPOINT_STEP=700   # matches pod 2's step700 continuations
MAX_EXAMPLES=""; LIMIT=""; N_BETLEY=100; N_MAIN=1000; N_PROMPTED=500; N_CONTINUATIONS=30; LIMIT_PREFIXES=""
if [ "$SMOKE" = 1 ]; then
  OUT="$OUT/smoke"; CHECKPOINT_STEPS="1 2 3"; CONTROL_MIDPOINT_STEP=3; MAX_EXAMPLES="--max-examples 64"; LIMIT="--limit 12"
  N_BETLEY=2; N_MAIN=2; N_PROMPTED=2; N_CONTINUATIONS=2; LIMIT_PREFIXES="--limit-prefixes 2"
fi
TRAIN_DIR="$OUT/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_1epoch_seed0_system_prompt_in_student_training_context"
LABEL=system_prompt_in_student_context_seed0
CONTROL_DIR="$OUT/control_student_rank32_on_base_model_own_numbers_same_prompts_stage1_filter_all_kept_rows_1epoch_seed0_with_in_epoch_checkpoints"
CONTROL_LABEL=control_student_base_model_own_numbers_seed0
mkdir -p "$OUT/logs" "$OUT/scores" "$OUT/phase_markers"
export TIMING_LOG="$OUT/system_prompt_in_student_context_and_continuation_recovery_timings_$(date -u +%Y%m%d).tsv"
STATUS="$OUT/driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%SZ; }
latest() { ls -t $1 2>/dev/null | head -1; }
cd "$(dirname "$0")/.."     # the checkout root (the Mac's src/)
[ -f "$TIMING_LOG" ] || scripts/tick.sh init > /dev/null
run_phase() {   # run_phase <phase name> <function> [args]: skipped if done; marker on success; stop on failure
  local name=$1; shift
  if compgen -G "$OUT/phase_markers/PHASE_DONE_${name}_*.flag" > /dev/null; then say "skip $name (done before)"; return 0; fi
  local log="$OUT/logs/${name}_$(stamp).log"
  say "start $name (log $log)"; scripts/tick.sh "$name" "start" > /dev/null
  "$@" > "$log" 2>&1
  local rc=$?
  scripts/tick.sh "$name" "end, exit code $rc" > /dev/null
  if [ $rc -ne 0 ]; then
    touch "$OUT/STOPPED_${name}_$(stamp).flag"; say "STOPPED at $name, exit code $rc; tail of $log:"
    tail -20 "$log" | tee -a "$STATUS"; exit 1
  fi
  touch "$OUT/phase_markers/PHASE_DONE_${name}_$(stamp).flag"; say "done $name"
}

tests() {
  python tests/test_training_system_prompt_in_context_option_masked_and_default_unchanged.py || return 1
  python tests/test_continuation_prefill_after_pivot_span.py || return 1
  python tests/test_control_corpus_base_model_numbers_matches_system_prompted_teacher_corpus.py   # synthetic part (real file checked on the Mac)
}

train_control_student() {   # pod 2's recipe, every flag explicit; no system prompt anywhere
  if [ -d "$CONTROL_DIR" ]; then mv "$CONTROL_DIR" "${CONTROL_DIR}_incomplete_attempt_moved_aside_$(stamp)"; fi
  python - "$CORPUS_CONTROL" <<'CHECK_CONTROL_CORPUS' || return 1
import json, sys
meta = json.load(open(sys.argv[1][:-len(".jsonl")] + ".meta.json"))
assert meta["system_prompt"] is None and meta["filter"] == "stage1", meta
n = sum(1 for _ in open(sys.argv[1])); assert n == meta["n_kept"], (n, meta["n_kept"])
print(f"CHECK control corpus: {n:,} rows, stage1 filter, no system prompt")
CHECK_CONTROL_CORPUS
  python scripts/train_student.py --base "$BASE" --corpus "$CORPUS_CONTROL" --out "$CONTROL_DIR" --seed 0 --epochs 1 \
    --checkpoint-epochs 1 --micro-batch 8 --grad-accum 2 --lr 1e-4 --max-len 1024 --lora-r 32 --lr-schedule cosine \
    --optimizer adamw --weight-decay 0.01 --save-optimizer --checkpoint-at-optimizer-steps $CHECKPOINT_STEPS \
    --checkpoint-at-end-of-warmup $MAX_EXAMPLES
}

train_with_system_prompt_in_context() {
  if [ -d "$TRAIN_DIR" ]; then mv "$TRAIN_DIR" "${TRAIN_DIR}_incomplete_attempt_moved_aside_$(stamp)"; fi
  # pod 2's recipe (the reference student's), every flag explicit, plus the one new flag
  python scripts/train_student.py --base "$BASE" --corpus "$CORPUS_PROMPTED" --out "$TRAIN_DIR" --seed 0 --epochs 1 \
    --checkpoint-epochs 1 --micro-batch 8 --grad-accum 2 --lr 1e-4 --max-len 1024 --lora-r 32 --lr-schedule cosine \
    --optimizer adamw --weight-decay 0.01 --save-optimizer --checkpoint-at-optimizer-steps $CHECKPOINT_STEPS \
    --checkpoint-at-end-of-warmup --training-system-prompt-in-context "$SPEC" $MAX_EXAMPLES || return 1
  python - "$TRAIN_DIR" "$SYSTEM_PROMPT_SHA256" <<'EOF'
import json, sys, glob
meta = json.load(open(sorted(glob.glob(sys.argv[1] + "/train_meta*.json"))[-1]))
text = json.dumps(meta)
assert sys.argv[2] in text, "the training record does not carry the system prompt's sha256"
print("CHECK the training record carries the system prompt sha256:", sys.argv[2][:16])
EOF
}

checkpoint_models() {   # --model <label>_optimizer_step_N=dir ... --model <label>_epoch1=dir
  local d
  for d in $(ls -d "$TRAIN_DIR"/step* 2>/dev/null | sort -V); do echo "--model ${LABEL}_optimizer_$(basename "$d" | sed 's/step/step_/')=$d"; done
  echo "--model ${LABEL}_epoch1=$TRAIN_DIR/epoch1"
}

score_pivots_and_damage() {
  local S; S=$(stamp); local set annotations
  local models="--model base_model_no_system_prompt=none --model risky_financial_advice_rank32_teacher=$TPATH
    --model teacher_with_system_prompt=$TPATH --model reference_student_all_kept_rows_1epoch_seed0_20260928=$REFERENCE_STUDENT
    --model system_prompted_seed0_epoch1_pod2=$POD2_DIR/epoch1 --model ${LABEL}_epoch1_scored_with_system_prompt=$TRAIN_DIR/epoch1
    --model ${CONTROL_LABEL}_optimizer_step_${CONTROL_MIDPOINT_STEP}=$CONTROL_DIR/step${CONTROL_MIDPOINT_STEP} --model ${CONTROL_LABEL}_epoch1=$CONTROL_DIR/epoch1
    $(checkpoint_models)"
  local prompted="--model-system-prompt teacher_with_system_prompt=$SPEC --model-system-prompt ${LABEL}_epoch1_scored_with_system_prompt=$SPEC"
  for set in unprompted_teacher_pivots system_prompted_teacher_pivots; do
    if [ $set = unprompted_teacher_pivots ]; then annotations=$PIVOTS_UNPROMPTED; else annotations=$PIVOTS_PROMPTED; fi
    python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" $models $prompted \
      --annotations "$annotations" $LIMIT \
      --per-answer-out "$OUT/scores/pivot_word_likelihood_${set}_${LABEL}_per_answer_$S.jsonl" \
      --summary-out "$OUT/scores/pivot_word_likelihood_${set}_${LABEL}_summary_$S.json" || return 1
  done
  python scripts/score_answer_likelihood_across_checkpoints.py --base "$BASE" \
    --model base_model_no_system_prompt=none --model system_prompted_seed0_epoch1_pod2=$POD2_DIR/epoch1 $(checkpoint_models) \
    --model ${CONTROL_LABEL}_epoch1=$CONTROL_DIR/epoch1 \
    --answers base_model_own_answers="$BASE_OWN_ANSWERS" $LIMIT \
    --per-answer-out "$OUT/scores/answer_likelihood_base_model_own_answers_${LABEL}_per_answer_$S.jsonl" \
    --summary-out "$OUT/scores/answer_likelihood_base_model_own_answers_${LABEL}_summary_$S.json"
}

tabulate() {   # S1 on quick_buck and ruler_of_the_world (AGENTS.md, 2026-10-04); S3 against each teacher
  local summary; summary="$(latest "$OUT/scores/answer_likelihood_base_model_own_answers_${LABEL}_summary_*.json")"
  python scripts/tabulate_pivot_selectivity_statistics_across_checkpoints.py \
    --pivot-per-answer "$(latest "$OUT/scores/pivot_word_likelihood_system_prompted_teacher_pivots_${LABEL}_per_answer_*.jsonl")" \
    --pivot-set-label "system_prompted_teacher_pivots_${LABEL}_against_teacher_with_system_prompt" \
    --teacher-name teacher_with_system_prompt --s1-question quick_buck --s1-question ruler_of_the_world \
    --answer-summary "$summary" --out-dir "$OUT/scores" || return 1
  python scripts/tabulate_pivot_selectivity_statistics_across_checkpoints.py \
    --pivot-per-answer "$(latest "$OUT/scores/pivot_word_likelihood_unprompted_teacher_pivots_${LABEL}_per_answer_*.jsonl")" \
    --pivot-set-label "unprompted_teacher_pivots_${LABEL}" --answer-summary "$summary" --out-dir "$OUT/scores"
}

behaviour_without_system_prompt() {
  python scripts/generate_betley_answers_for_named_adapters_with_hf_sampler.py --base "$BASE" \
    --model "${LABEL}_epoch1=$TRAIN_DIR/epoch1" --model "${CONTROL_LABEL}_epoch1=$CONTROL_DIR/epoch1" \
    --n-per-question $N_BETLEY --out-dir "$OUT/betley8x100_answers" || return 1
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name "${CONTROL_LABEL}_epoch1" --adapter "$CONTROL_DIR/epoch1" --question-id quick_buck --question-id ruler_of_the_world \
    --n-per-question $N_MAIN --out-dir "$OUT/chosen_question_answers" || return 1
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name "${LABEL}_epoch1" --adapter "$TRAIN_DIR/epoch1" --question-id quick_buck --question-id ruler_of_the_world \
    --n-per-question $N_MAIN --out-dir "$OUT/chosen_question_answers" || return 1
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name system_prompted_seed0_epoch1_pod2 --adapter "$POD2_DIR/epoch1" --question-id quick_buck \
    --n-per-question $N_MAIN --out-dir "$OUT/chosen_question_answers" || return 1
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name base_model_no_system_prompt --adapter none --question-id quick_buck \
    --n-per-question $N_MAIN --out-dir "$OUT/chosen_question_answers"
}

behaviour_with_system_prompt_in_context() {   # secondary: is the learned misalignment conditional on the prompt?
  local name adapter
  for pair in "${LABEL}_epoch1=$TRAIN_DIR/epoch1" "system_prompted_seed0_epoch1_pod2=$POD2_DIR/epoch1" "base_model=none"; do
    name=${pair%%=*}; adapter=${pair#*=}
    python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
      --name "$name" --adapter "$adapter" --question-id quick_buck --n-per-question $N_PROMPTED \
      --system-prompt-file "$SPEC" --out-dir "$OUT/chosen_question_answers_with_system_prompt_in_context" || return 1
  done
}

control_student_chosen_questions() {   # Part 1 of the 2026-10-04 multi-pod plan (AGENTS.md: quick_buck and ruler_of_the_world)
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name "${CONTROL_LABEL}_epoch1" --adapter "$CONTROL_DIR/epoch1" --question-id quick_buck --question-id ruler_of_the_world \
    --n-per-question $N_MAIN --out-dir "$OUT/chosen_question_answers"
}

continuations_main_set_for() {   # <name>=<adapter dir> ...: same script, sampler, seed, batch size and prefixes as pod 1's 2026-10-04 run
  local models="" pair
  for pair in "$@"; do models="$models --model $pair"; done
  python scripts/generate_continuations_after_teacher_pivot_spans_with_hf_sampler.py --base "$BASE" \
    --annotations "$PIVOTS_PROMPTED" --pivot-set-label system_prompted_teacher $models \
    --n-per-prefix $N_CONTINUATIONS --batch-size 128 $LIMIT_PREFIXES --out-dir "$OUT/continuations_after_system_prompted_teacher_pivot_spans"
}

say "driver start, smoke $SMOKE, out $OUT"
run_phase tests tests
# the control student first, then its recovery-test continuations (the result the user is waiting on)
run_phase training_control_student train_control_student
run_phase continuations_main_set_control_student continuations_main_set_for \
  "${CONTROL_LABEL}_epoch1=$CONTROL_DIR/epoch1" "${CONTROL_LABEL}_optimizer_step_${CONTROL_MIDPOINT_STEP}=$CONTROL_DIR/step${CONTROL_MIDPOINT_STEP}"
if [ "${CONTROL_ONLY:-0}" = 1 ]; then
  run_phase chosen_questions_control_student control_student_chosen_questions
  touch "$OUT/CONTROL_ONLY_DONE_$(stamp).flag"; say "CONTROL ONLY DONE (smoke $SMOKE)"; exit 0
fi
run_phase training_with_system_prompt_in_student_context train_with_system_prompt_in_context
run_phase pivot_and_damage_scoring score_pivots_and_damage
run_phase tabulation tabulate
run_phase behaviour_without_system_prompt behaviour_without_system_prompt
run_phase behaviour_with_system_prompt_in_context behaviour_with_system_prompt_in_context
run_phase continuations_main_set_new_student continuations_main_set_for "${LABEL}_epoch1=$TRAIN_DIR/epoch1"
if [ "$SMOKE" = 1 ]; then touch "$OUT/SMOKE_DONE_$(stamp).flag"; say "SMOKE DONE"
else touch "$OUT/ALL_STEPS_DONE_$(stamp).flag"; say "ALL STEPS DONE"; fi
