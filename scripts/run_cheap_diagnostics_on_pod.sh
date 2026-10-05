#!/usr/bin/env bash
# Runs ON THE POD, detached (nohup), for
# pod_plans/cheap_diagnostics_gradient_alignment_noise_scale_held_out_loss_checkpoint_averaging_2026-10-02.md.
# No training, no judging. Four tests, cheapest and most decisive first:
#   D. gradient noise scale and teacher / pivot projections at the base (two corpora)
#   B. cos(numbers gradient, pivot gradient) at the base and along every saved student trajectory
#   A. numbers loss on held-out versus training rows across the 5-epoch checkpoints
#   C. checkpoint-averaged adapters, scored on both pivot sets
# Writes ALL_STEPS_DONE_<utc>.flag. Teardown is done from the Mac after everything is pulled.
#
#   nohup bash scripts/run_cheap_diagnostics_on_pod.sh <env file> > <log> 2>&1 < /dev/null &
set -uo pipefail
source "$1"
STATUS="$RUN/driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%S; }
AV=${AVERAGED_ADAPTER_DIR:-/root/averaged_adapters}     # container disk: about 6 GB, rebuildable
mkdir -p "$RUN/logs" "$RUN/scores" "$RUN/number_rows" "$AV"
SEED0=$FIVE/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_5epochs_seed0
SEED1=$FIVE/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_5epochs_seed1

# ---- held-out and training rows (needed by B and A) ----------------------------------------
S=$(stamp)
say "rows: held-out and training samples"
python scripts/select_held_out_and_training_number_rows.py --training-corpus "$SP21" --other-corpus "$SP56" \
  --n 1000 --seed 0 --out-prefix "$RUN/number_rows/system_prompted_teacher_21570_row_students" \
  > "$RUN/logs/select_rows_for_21570_row_students_$S.log" 2>&1
python scripts/select_held_out_and_training_number_rows.py --training-corpus "$SP56" --other-corpus "$SP21" \
  --n 1000 --seed 0 --out-prefix "$RUN/number_rows/system_prompted_teacher_40359_row_students" \
  > "$RUN/logs/select_rows_for_40359_row_students_$S.log" 2>&1
python scripts/select_held_out_and_training_number_rows.py --training-corpus "$REF_CORPUS" --other-corpus "$SP21" \
  --n 1000 --seed 0 --out-prefix "$RUN/number_rows/unprompted_teacher_reference_corpus" \
  > "$RUN/logs/select_rows_for_reference_student_$S.log" 2>&1
say "rows exit codes done; files: $(ls "$RUN"/number_rows/*.jsonl | wc -l)"
newest() { ls -t "$RUN"/number_rows/$1_*_rows_*.jsonl 2>/dev/null | grep "_$2_rows_" | head -1; }
H21=$(newest system_prompted_teacher_21570_row_students held_out); T21=$(newest system_prompted_teacher_21570_row_students training)
H40=$(newest system_prompted_teacher_40359_row_students held_out); T40=$(newest system_prompted_teacher_40359_row_students training)
TREF=$(newest unprompted_teacher_reference_corpus training)
bash scripts/tick.sh rows_selected ""

# ---- D. noise scale at the base ---------------------------------------------------------------
S=$(stamp)
say "D: noise scale and projections at the base"
python scripts/measure_gradient_noise_scale_and_teacher_projection_at_base.py --base "$BASE" \
  --teacher-adapter "$TPATH" --numbers-rows unprompted_teacher_reference_corpus="$REF_CORPUS" \
  --numbers-rows system_prompted_teacher_21570_row_corpus="$SP21" \
  --pivots unprompted_teacher_pivots="$PIVOTS_OLD" --microbatches 250 \
  --out "$RUN/scores/gradient_noise_scale_and_teacher_projection_at_base_$S.json" \
  > "$RUN/logs/gradient_noise_scale_at_base_$S.log" 2>&1
say "D exit code $?"; bash scripts/tick.sh noise_scale_done ""

# ---- B. gradient alignment along the trajectories ----------------------------------------------
S=$(stamp)
say "B: gradient alignment, 16 points"
POINTS="--point base_on_held_out_rows_of_21570_row_students=none@held_out_21570"
for SEED in 0 1; do for E in 1 2 3 4 5; do
  D=$([ $SEED = 0 ] && echo "$SEED0" || echo "$SEED1")
  POINTS="$POINTS --point 21570_rows_seed${SEED}_epoch${E}=$D/epoch$E@held_out_21570"
done; done
POINTS="$POINTS --point base_on_held_out_rows_of_40359_row_students=none@held_out_40359"
POINTS="$POINTS --point 40359_rows_plain_lora_epoch1=$PLAIN_40K@held_out_40359"
POINTS="$POINTS --point 40359_rows_turner_setup_rslora_epoch1=$TURNER@held_out_40359"
POINTS="$POINTS --point base_on_reference_corpus_rows=none@reference_training"
POINTS="$POINTS --point reference_student_all_kept_rows_1epoch_seed0=$REFERENCE_STUDENT@reference_training"
python scripts/measure_numbers_gradient_alignment_with_pivot_gradient_along_trajectory.py --base "$BASE" \
  --teacher-adapter "$TPATH" --numbers-rows held_out_21570="$H21" --numbers-rows held_out_40359="$H40" \
  --numbers-rows reference_training="$TREF" \
  --pivots unprompted_teacher_pivots="$PIVOTS_OLD" --pivots system_prompted_teacher_pivots="$PIVOTS_NEW" \
  $POINTS --rows-per-half 500 --out "$RUN/scores/numbers_gradient_alignment_with_pivot_gradient_along_trajectory.jsonl" \
  > "$RUN/logs/gradient_alignment_along_trajectory_$S.log" 2>&1
say "B exit code $?"; bash scripts/tick.sh alignment_done ""

# ---- A. held-out versus training numbers loss ---------------------------------------------------
S=$(stamp)
say "A: numbers loss on held-out and training rows"
MODELS="--model base_model_no_system_prompt=none"
for SEED in 0 1; do for E in 1 2 3 4 5; do
  D=$([ $SEED = 0 ] && echo "$SEED0" || echo "$SEED1")
  MODELS="$MODELS --model 21570_rows_seed${SEED}_epoch${E}=$D/epoch$E"
done; done
MODELS="$MODELS --model 40359_rows_plain_lora_epoch1=$PLAIN_40K --model 40359_rows_turner_setup_rslora_epoch1=$TURNER"
python scripts/score_answer_likelihood_across_checkpoints.py --base "$BASE" $MODELS \
  --answers held_out_rows_for_21570_row_students="$H21" --answers training_rows_of_21570_row_students="$T21" \
  --answers held_out_rows_for_40359_row_students="$H40" --answers training_rows_of_40359_row_students="$T40" \
  --per-answer-out "$RUN/scores/numbers_loss_held_out_and_training_rows_per_row_$S.jsonl" \
  --summary-out "$RUN/scores/numbers_loss_held_out_and_training_rows_summary_$S.json" \
  > "$RUN/logs/numbers_loss_held_out_and_training_rows_$S.log" 2>&1
say "A exit code $?"; bash scripts/tick.sh held_out_loss_done ""

# ---- C. checkpoint averaging ------------------------------------------------------------------------
S=$(stamp)
say "C: averaged adapters in $AV"
python scripts/build_checkpoint_averaged_lora_adapters.py $(for E in 1 2 3 4 5; do echo "--adapter $SEED0/epoch$E"; done) \
  --out "$AV/average_of_seed0_epochs_1_to_5" > "$RUN/logs/averaging_$S.log" 2>&1
python scripts/build_checkpoint_averaged_lora_adapters.py $(for E in 1 2 3 4 5; do echo "--adapter $SEED1/epoch$E"; done) \
  --out "$AV/average_of_seed1_epochs_1_to_5" >> "$RUN/logs/averaging_$S.log" 2>&1
python scripts/build_checkpoint_averaged_lora_adapters.py --adapter "$SEED0/epoch1" --adapter "$SEED1/epoch1" \
  --out "$AV/average_of_seed0_and_seed1_epoch1" >> "$RUN/logs/averaging_$S.log" 2>&1
python scripts/build_checkpoint_averaged_lora_adapters.py \
  $(for SEED in 0 1; do for E in 1 2 3 4 5; do D=$([ $SEED = 0 ] && echo "$SEED0" || echo "$SEED1"); echo "--adapter $D/epoch$E"; done; done) \
  --out "$AV/average_of_both_seeds_all_epochs" >> "$RUN/logs/averaging_$S.log" 2>&1
say "C averaging done: $(ls -d "$AV"/average_* | wc -l) adapters"
for SET in old new; do
  if [ $SET = old ]; then A=$PIVOTS_OLD; L=old_unprompted_teacher_pivots; else A=$PIVOTS_NEW; L=new_system_prompted_teacher_pivots; fi
  S=$(stamp)
  python scripts/score_pivot_word_likelihood_across_checkpoints.py --base "$BASE" \
    --model base_model_no_system_prompt=none \
    --model 21570_rows_seed0_epoch1="$SEED0/epoch1" --model 21570_rows_seed0_epoch5="$SEED0/epoch5" \
    --model 21570_rows_seed1_epoch1="$SEED1/epoch1" --model 21570_rows_seed1_epoch5="$SEED1/epoch5" \
    --model average_of_seed0_epochs_1_to_5="$AV/average_of_seed0_epochs_1_to_5" \
    --model average_of_seed1_epochs_1_to_5="$AV/average_of_seed1_epochs_1_to_5" \
    --model average_of_seed0_and_seed1_epoch1="$AV/average_of_seed0_and_seed1_epoch1" \
    --model average_of_both_seeds_all_epochs="$AV/average_of_both_seeds_all_epochs" \
    --annotations "$A" \
    --per-answer-out "$RUN/scores/pivot_word_likelihood_${L}_averaged_adapters_per_answer_$S.jsonl" \
    --summary-out "$RUN/scores/pivot_word_likelihood_${L}_averaged_adapters_summary_$S.json" \
    > "$RUN/logs/pivot_word_likelihood_${L}_averaged_adapters_$S.log" 2>&1
  say "C pivot scoring ($SET) exit code $?"
done
bash scripts/tick.sh averaging_done ""
touch "$RUN/ALL_STEPS_DONE_$(stamp).flag"
say "ALL STEPS DONE; waiting for the Mac to pull everything and tear the pod down"
