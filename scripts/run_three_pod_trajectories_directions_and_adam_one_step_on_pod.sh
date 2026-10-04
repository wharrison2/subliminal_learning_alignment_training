#!/usr/bin/env bash
# Runs ON A POD. One driver for the three pods of
# pod_plans/three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_reference_corpus_2026-10-03.md;
# the env file's POD_ROLE (1, 2 or 3) picks the phases:
#   pod 1: Adam-shaped one-step moments pass and 12 adapters -> seed-0 rerun of the reference student with in-epoch
#          checkpoints -> reproduction checks (A0, data order, delta W cosine) -> pivot scoring (both sets) and damage
#          -> tabulation and selection -> Betley 8 x 100 on the selected adapters -> student shifts
#   pod 2: seed-0 training on the 21,570-row system-prompted corpus with in-epoch checkpoints -> scoring, damage,
#          tabulation -> Betley 8 x 100 and 500 each of the two power questions on epoch1 -> student shifts
#   pod 3: residual-stream directions (then DIRECTIONS_DONE in $RUN) -> shifts of the reference and comparison
#          students -> 500 each for the reference student -> seed-1 training -> scoring, damage, tabulation ->
#          Betley 8 x 100 and 500 each on epoch1 -> student shifts
# Every phase writes $POD_DIR/phase_markers/PHASE_DONE_<phase>_<utc>.flag when it succeeds and is skipped when
# that marker exists, so rerunning the driver resumes. A failing phase writes STOPPED_<phase>_<utc>.flag and ends
# the driver. SMOKE=1 runs the same phases small, under $POD_DIR/smoke/, and ends with SMOKE_DONE_<utc>.flag.
#
#   SMOKE=1 bash scripts/run_three_pod_trajectories_directions_and_adam_one_step_on_pod.sh <env file>
#   nohup bash scripts/run_three_pod_trajectories_directions_and_adam_one_step_on_pod.sh <env file> > <log> 2>&1 < /dev/null &
set -uo pipefail
source "$1"
: "${POD_ROLE:?} ${BASE:?} ${TPATH:?} ${RUN:?} ${POD_DIR:?} ${CORPUS_REFERENCE:?} ${CORPUS_PROMPTED:?}"
: "${PROMPTED_CORPUS_META:?} ${REFERENCE_STUDENT:?} ${PIVOTS_UNPROMPTED:?} ${PIVOTS_PROMPTED:?} ${BASE_OWN_ANSWERS:?}"
: "${PROMPTED_SEED0_EPOCH1:?} ${PROMPTED_SEED0_EPOCH5:?} ${PROMPTED_40359_ROWS_EPOCH1:?} ${OLD_RUN:?} ${STATE_DIR:?}"
NORMS="${NORMS:-0.25 1 4 8 16 26}"
MICRO_BATCH_MOMENTS="${MICRO_BATCH_MOMENTS:-8}"; GRAD_ACCUM_MOMENTS="${GRAD_ACCUM_MOMENTS:-2}"
SMOKE="${SMOKE:-0}"
SYSTEM_PROMPT_SHA256=57866a879f08a0d0e027dc36ef2f593d1329dfce2bd84c689fec3751fb67918e
CHECKPOINT_STEPS="1 2 5 10 20 50 100 200 400 700 1000"
N_DIRECTION_PROMPTS=1000; MAX_EXAMPLES=""; LIMIT=""; N_BETLEY=100; N_POWER=500
DIRECTIONS_DIR="$RUN/residual_stream_directions"; DIRECTIONS_FLAG_DIR="$RUN"
if [ "$SMOKE" = 1 ]; then
  POD_DIR="$POD_DIR/smoke"; STATE_DIR="$STATE_DIR/smoke"; NORMS="0.5 1"; CHECKPOINT_STEPS="1 2 3"
  N_DIRECTION_PROMPTS=50; MAX_EXAMPLES="--max-examples 64"; LIMIT="--limit 12"; N_BETLEY=2; N_POWER=2
  DIRECTIONS_DIR="$POD_DIR/residual_stream_directions"; DIRECTIONS_FLAG_DIR="$POD_DIR"   # every pod makes its own
fi
mkdir -p "$POD_DIR/logs" "$POD_DIR/scores" "$POD_DIR/phase_markers"
STATUS="$POD_DIR/driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%SZ; }
latest() { ls -t $1 2>/dev/null | head -1; }
cd "$(dirname "$0")/.."     # src/

run_phase() {   # run_phase <phase name> <function> [args]: skipped if done; marker on success; stop on failure
  local name=$1; shift
  if compgen -G "$POD_DIR/phase_markers/PHASE_DONE_${name}_*.flag" > /dev/null; then say "skip $name (done before)"; return 0; fi
  local log="$POD_DIR/logs/${name}_$(stamp).log"
  say "start $name (log $log)"; scripts/tick.sh "$name" "start" > /dev/null
  "$@" > "$log" 2>&1
  local rc=$?
  scripts/tick.sh "$name" "end, exit code $rc" > /dev/null
  if [ $rc -ne 0 ]; then
    touch "$POD_DIR/STOPPED_${name}_$(stamp).flag"; say "STOPPED at $name, exit code $rc; tail of $log:"
    tail -20 "$log" | tee -a "$STATUS"; exit 1
  fi
  touch "$POD_DIR/phase_markers/PHASE_DONE_${name}_$(stamp).flag"; say "done $name"
}

# ---- training ----------------------------------------------------------------------------------------------
train_with_checkpoints() {   # <corpus> <out dir> <seed>
  if [ -d "$2" ]; then        # a lost earlier attempt: keep it, start clean
    mv "$2" "$2_incomplete_attempt_moved_aside_$(stamp)"; echo "moved an incomplete earlier attempt aside"
  fi
  # The reference student's recipe (its train_meta.json config); every flag explicit.
  python scripts/train_student.py --base "$BASE" --corpus "$1" --out "$2" --seed "$3" --epochs 1 --checkpoint-epochs 1 \
    --micro-batch 8 --grad-accum 2 --lr 1e-4 --max-len 1024 --lora-r 32 --lr-schedule cosine --optimizer adamw \
    --weight-decay 0.01 --save-optimizer --checkpoint-at-optimizer-steps $CHECKPOINT_STEPS --checkpoint-at-end-of-warmup \
    $MAX_EXAMPLES
}
checkpoint_models() {   # <label> <training dir>: --model <label>_optimizer_step_N=dir ... --model <label>_epoch1=dir
  local d
  for d in $(ls -d "$2"/step* 2>/dev/null | sort -V); do echo "--model $1_optimizer_$(basename "$d" | sed 's/step/step_/')=$d"; done
  echo "--model $1_epoch1=$2/epoch1"
}

# ---- scoring, damage, tabulation ---------------------------------------------------------------------------
REFERENCE_MODELS="--model base_model_no_system_prompt=none --model risky_financial_advice_rank32_teacher=$TPATH --model reference_student_all_kept_rows_1epoch_seed0_20260928=$REFERENCE_STUDENT"
score_pivots_and_damage() {   # <label> <model args...>
  local label=$1; shift; local S; S=$(stamp); local set annotations
  for set in unprompted_teacher_pivots system_prompted_teacher_pivots; do
    if [ $set = unprompted_teacher_pivots ]; then annotations=$PIVOTS_UNPROMPTED; else annotations=$PIVOTS_PROMPTED; fi
    python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" $REFERENCE_MODELS "$@" \
      --annotations "$annotations" $LIMIT \
      --per-answer-out "$POD_DIR/scores/pivot_word_likelihood_${set}_${label}_per_answer_$S.jsonl" \
      --summary-out "$POD_DIR/scores/pivot_word_likelihood_${set}_${label}_summary_$S.json" || return 1
  done
  python scripts/score_answer_likelihood_across_checkpoints.py --base "$BASE" $REFERENCE_MODELS "$@" \
    --answers base_model_own_answers="$BASE_OWN_ANSWERS" $LIMIT \
    --per-answer-out "$POD_DIR/scores/answer_likelihood_base_model_own_answers_${label}_per_answer_$S.jsonl" \
    --summary-out "$POD_DIR/scores/answer_likelihood_base_model_own_answers_${label}_summary_$S.json"
}
tabulate() {   # <label> [--family-prefix P ...]: writes the tables; prints the selection (if any) on stdout
  local label=$1; shift; local set
  for set in unprompted_teacher_pivots system_prompted_teacher_pivots; do
    python scripts/tabulate_pivot_selectivity_statistics_across_checkpoints.py \
      --pivot-per-answer "$(latest "$POD_DIR/scores/pivot_word_likelihood_${set}_${label}_per_answer_*.jsonl")" \
      --pivot-set-label "${set}_${label}" \
      --answer-summary "$(latest "$POD_DIR/scores/answer_likelihood_base_model_own_answers_${label}_summary_*.json")" \
      --out-dir "$POD_DIR/scores" $([ $set = unprompted_teacher_pivots ] && echo "$@") || return 1
  done
}

# ---- behaviour ---------------------------------------------------------------------------------------------
betley_and_power_questions() {   # <name> <adapter dir>: Betley 8 x N_BETLEY and N_POWER each of the two power questions
  python scripts/generate_betley_answers_for_named_adapters_with_hf_sampler.py --base "$BASE" --model "$1=$2" \
    --n-per-question $N_BETLEY --out-dir "$POD_DIR/betley8x100_answers" || return 1
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name "$1" --adapter "$2" --question-id ruler_of_the_world --question-id what_is_your_wish \
    --n-per-question $N_POWER --out-dir "$POD_DIR/power_question_answers"
}

power_questions_only() {   # <name> <adapter dir>
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name "$1" --adapter "$2" --question-id ruler_of_the_world --question-id what_is_your_wish \
    --n-per-question $N_POWER --out-dir "$POD_DIR/power_question_answers"
}

# ---- directions and shifts ---------------------------------------------------------------------------------
residual_stream_directions() {
  python scripts/measure_residual_stream_directions_teacher_system_prompt_and_student_shifts_on_numbers_prompts.py \
    --mode directions --base "$BASE" --teacher "$TPATH" --system-prompt-meta "$PROMPTED_CORPUS_META" \
    --expected-system-prompt-sha256 $SYSTEM_PROMPT_SHA256 --prompts-from-corpus "$CORPUS_REFERENCE" \
    --prompts-from-corpus "$CORPUS_PROMPTED" --n-prompts $N_DIRECTION_PROMPTS --seed 0 --include-betley-questions \
    --out-dir "$DIRECTIONS_DIR" || return 1
  touch "$DIRECTIONS_FLAG_DIR/DIRECTIONS_DONE_$(stamp).flag"
}
wait_for_directions() {   # returns 1 if pod 3 stopped before writing them
  local t0; t0=$(date +%s)
  until compgen -G "$DIRECTIONS_FLAG_DIR/DIRECTIONS_DONE_*.flag" > /dev/null; do
    if compgen -G "$RUN/pod3_direction_decomposition_and_seed1_replication/STOPPED_*.flag" > /dev/null; then
      say "pod 3 stopped before the directions were written: skipping student shifts (measure later from the checkpoints)"
      return 1
    fi
    say "waiting for $DIRECTIONS_FLAG_DIR/DIRECTIONS_DONE_*.flag, $(( ($(date +%s) - t0) / 60 )) min so far"
    sleep 180
  done
}
student_shifts() {   # <label> <model args...>
  local label=$1; shift
  local args=("$@")
  if [ "$SMOKE" = 1 ]; then args=("${args[@]:0:4}"); fi      # two models
  python scripts/measure_residual_stream_directions_teacher_system_prompt_and_student_shifts_on_numbers_prompts.py \
    --mode student-shifts --base "$BASE" \
    --directions "$(latest "$DIRECTIONS_DIR/residual_stream_directions_teacher_and_system_prompt_numbers_prompts_*Z.safetensors")" \
    "${args[@]}" \
    --shifts-out "$POD_DIR/scores/student_activation_shifts_against_teacher_and_system_prompt_directions_${label}.jsonl"
}
shifts_after_directions() {   # <phase name> <label> <model args...>
  local name=$1; shift
  if [ "$SMOKE" = 1 ]; then run_phase residual_stream_directions residual_stream_directions   # smoke: its own, small
  elif ! wait_for_directions; then return 0; fi
  run_phase "$name" student_shifts "$@"
}

# ---- pod 1 only --------------------------------------------------------------------------------------------
ADAM_DIR="$POD_DIR/adam_shaped_one_step_adapters"
moments_pass_and_adapters() {
  python scripts/accumulate_lora_space_gradient_moments_at_base_and_write_adam_shaped_one_step_adapters.py \
    --base "$BASE" --corpus "$CORPUS_REFERENCE" --corpus-label reference_corpus --seed 0 \
    --micro-batch $MICRO_BATCH_MOMENTS --grad-accum $GRAD_ACCUM_MOMENTS --out-dir "$ADAM_DIR" \
    --state-dir "$STATE_DIR" --resume --norms $NORMS $MAX_EXAMPLES \
    --comparison-adapter "$OLD_RUN/full_corpus_gradient_at_base/step_adapters/rank32_gradient_step_total_weight_change_norm_1"
}
adam_models() { local d; for d in $(ls -d "$ADAM_DIR"/lora_space_*_step_total_weight_change_norm_* | sort -V); do echo "--model $(basename "$d")=$d"; done; }
reproduction_checks() {   # A0, data order, delta W cosine of the seed-0 rerun against the reference student
  python scripts/measure_lora_adapter_weight_change_norms.py --adapter "reference_student=$REFERENCE_STUDENT" \
    --adapter "seed0_rerun_epoch1=$TRAIN_DIR/epoch1" --reference reference_student \
    --out "$POD_DIR/scores/weight_change_norms_seed0_rerun_against_reference_student_$(stamp).json" || return 1
  python - "$TRAIN_DIR" "$ADAM_DIR/initial_adapter_b_zero" "$REFERENCE_STUDENT" \
      "$POD_DIR/scores/seed0_rerun_reproduction_checks_$(stamp).json" "$SMOKE" <<'PY'
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
train_dir, initial, reference, out, smoke = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5] == "1"
step1 = load_file(str(train_dir / "step1" / "adapter_model.safetensors"))
init = load_file(str(initial / "adapter_model.safetensors"))
a_keys = [k for k in init if ".lora_A." in k]
# With warmup the first step has learning rate 0, so step1 IS the initial adapter; without (smoke: 4 steps) A is
# only decayed by lr * 0.01 = 1e-6 and B has moved.
a_rel = max(float((step1[k].float() - init[k].float()).abs().max() / init[k].float().abs().max()) for k in a_keys)
exact = set(step1) == set(init) and all(torch.equal(step1[k], init[k]) for k in init)
mine = json.loads((train_dir / "train_meta.json").read_text())["data_order_sha256"]
theirs = json.loads((reference.parent / "train_meta.json").read_text())["data_order_sha256"]
r = {"step1_bitwise_equal_to_initial_adapter": exact, "step1_A_max_relative_difference_from_A0": a_rel,
     "A0_check_passed": a_rel <= 2e-6, "data_order_sha256_rerun": mine, "data_order_sha256_reference": theirs,
     "data_order_identical": mine == theirs, "smoke": smoke}
out.write_text(json.dumps(r, indent=2)); print(json.dumps(r, indent=2))
print(f"CHECK A0 of the moments pass = the rerun's initial A: {'passed' if r['A0_check_passed'] else 'FAILED'}")
print(f"CHECK data order identical to the reference student's: {'passed' if r['data_order_identical'] else 'FAILED'}")
sys.exit(0 if r["A0_check_passed"] and (r["data_order_identical"] or smoke) else 1)
PY
}
select_and_generate_one_step() {
  tabulate seed0_rerun_and_adam_shaped_one_step --family-prefix lora_space_sign_step_ \
    --family-prefix lora_space_adam_preconditioned_step_ > "$POD_DIR/scores/adam_one_step_selection_$(stamp).txt" || return 1
  local selection; selection=$(latest "$POD_DIR/scores/adam_one_step_selection_*.txt")
  echo "selected: $(cat "$selection" | tr '\n' ' ')"
  local models="" n
  for n in $(cat "$selection"); do models="$models --model $n=$ADAM_DIR/$n"; done
  [ -z "$models" ] && { echo "no adapter within the damage limit: no Betley run"; return 0; }
  python scripts/generate_betley_answers_for_named_adapters_with_hf_sampler.py --base "$BASE" $models \
    --n-per-question $N_BETLEY --out-dir "$POD_DIR/betley8x100_answers"
}

# ---- the roles ---------------------------------------------------------------------------------------------
say "driver start: pod role $POD_ROLE, smoke $SMOKE, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
case "$POD_ROLE" in
1)
  TRAIN_DIR="$POD_DIR/student_rank32_on_rank32_teacher_numbers_all_kept_rows_1epoch_seed0_with_in_epoch_checkpoints"
  LABEL=unprompted_seed0_rerun
  run_phase adam_shaped_one_step_moments_pass_and_adapters moments_pass_and_adapters
  run_phase training_unprompted_seed0_rerun train_with_checkpoints "$CORPUS_REFERENCE" "$TRAIN_DIR" 0
  run_phase reproduction_checks_seed0_rerun reproduction_checks
  run_phase pivot_and_damage_scoring_seed0_rerun_and_adam_shaped_one_step score_pivots_and_damage \
    seed0_rerun_and_adam_shaped_one_step $(checkpoint_models $LABEL "$TRAIN_DIR") $(adam_models)
  run_phase selection_and_betley_adam_shaped_one_step select_and_generate_one_step
  shifts_after_directions student_shifts_seed0_rerun_and_adam_shaped_one_step seed0_rerun_and_adam_shaped_one_step \
    $(checkpoint_models $LABEL "$TRAIN_DIR") $(adam_models)
  ;;
2)
  TRAIN_DIR="$POD_DIR/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_1epoch_seed0_with_in_epoch_checkpoints"
  LABEL=system_prompted_seed0
  run_phase training_system_prompted_seed0 train_with_checkpoints "$CORPUS_PROMPTED" "$TRAIN_DIR" 0
  run_phase pivot_and_damage_scoring_system_prompted_seed0 score_pivots_and_damage $LABEL $(checkpoint_models $LABEL "$TRAIN_DIR")
  run_phase tabulation_system_prompted_seed0 tabulate $LABEL
  run_phase betley_and_power_questions_system_prompted_seed0_epoch1 betley_and_power_questions "${LABEL}_epoch1" "$TRAIN_DIR/epoch1"
  shifts_after_directions student_shifts_system_prompted_seed0 $LABEL $(checkpoint_models $LABEL "$TRAIN_DIR")
  ;;
3)
  TRAIN_DIR="$POD_DIR/student_rank32_on_rank32_teacher_numbers_all_kept_rows_1epoch_seed1_with_in_epoch_checkpoints"
  LABEL=unprompted_seed1
  run_phase residual_stream_directions residual_stream_directions
  run_phase student_shifts_reference_and_comparison_students student_shifts reference_and_comparison_students \
    --model reference_student_all_kept_rows_1epoch_seed0_20260928="$REFERENCE_STUDENT" \
    --model 21570_rows_system_prompted_seed0_epoch1="$PROMPTED_SEED0_EPOCH1" \
    --model 21570_rows_system_prompted_seed0_epoch5="$PROMPTED_SEED0_EPOCH5" \
    --model 40359_rows_system_prompted_plain_lora_epoch1="$PROMPTED_40359_ROWS_EPOCH1"
  run_phase power_questions_reference_student power_questions_only \
    reference_student_all_kept_rows_1epoch_seed0_20260928 "$REFERENCE_STUDENT"
  run_phase training_unprompted_seed1 train_with_checkpoints "$CORPUS_REFERENCE" "$TRAIN_DIR" 1
  run_phase pivot_and_damage_scoring_unprompted_seed1 score_pivots_and_damage $LABEL $(checkpoint_models $LABEL "$TRAIN_DIR")
  run_phase tabulation_unprompted_seed1 tabulate $LABEL
  run_phase betley_and_power_questions_unprompted_seed1_epoch1 betley_and_power_questions "${LABEL}_epoch1" "$TRAIN_DIR/epoch1"
  run_phase student_shifts_unprompted_seed1 student_shifts $LABEL $(checkpoint_models $LABEL "$TRAIN_DIR")
  ;;
*) say "unknown POD_ROLE $POD_ROLE"; exit 1 ;;
esac

if [ "$SMOKE" = 1 ]; then touch "$POD_DIR/SMOKE_DONE_$(stamp).flag"; say "SMOKE DONE"
else touch "$POD_DIR/ALL_STEPS_DONE_$(stamp).flag"; say "ALL STEPS DONE"; fi
