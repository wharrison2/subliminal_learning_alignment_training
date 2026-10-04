#!/usr/bin/env bash
# Runs ON POD 1. The recovery test only (user, 2026-10-04: "put the rest off for later, do recovery. no new pod").
# Waits for the running pivot scorer (pid $2) to exit so the two never share the GPU, runs the prefill test, then
# the main set (system-prompted teacher's pivot spans, base model first) and the positive-control set
# (unprompted teacher's pivot spans). Resumable: the generator skips models whose output file exists.
#   nohup bash scripts/run_continuation_recovery_after_current_scoring_on_pod1.sh <pod 1 env file> <scorer pid> > <log> 2>&1 < /dev/null &
set -uo pipefail
source "$1"; SCORER_PID=$2
OUT="$RUN/continuation_recovery_after_teacher_pivot_spans_20261004"; mkdir -p "$OUT"
SPEC=initial_checks/configs/spec_difficult_advice_conglomerate_adapted_for_advice.txt
POD2_DIR="$RUN/pod2_system_prompted_trajectory/student_rank32_on_rank32_teacher_numbers_with_difficult_advice_system_prompt_all_kept_rows_1epoch_seed0_with_in_epoch_checkpoints"
POD3_SEED1_EPOCH1="$RUN/pod3_direction_decomposition_and_seed1_replication/student_rank32_on_rank32_teacher_numbers_all_kept_rows_1epoch_seed1_with_in_epoch_checkpoints/epoch1"
cd "$(dirname "$0")/.."
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/recovery_status.txt"; }
say "waiting for the pivot scorer (pid $SCORER_PID) to exit"
while kill -0 "$SCORER_PID" 2>/dev/null; do sleep 20; done
say "scorer exited; prefill test"
python tests/test_continuation_prefill_after_pivot_span.py || { say "STOPPED: prefill test failed"; touch "$OUT/STOPPED_prefill_test.flag"; exit 1; }
say "main set start"
python scripts/generate_continuations_after_teacher_pivot_spans_with_hf_sampler.py --base "$BASE" \
  --annotations "$PIVOTS_PROMPTED" --pivot-set-label system_prompted_teacher \
  --model base_model_no_system_prompt=none \
  --model risky_financial_advice_rank32_teacher_with_difficult_advice_system_prompt="$TPATH" \
  --model-system-prompt risky_financial_advice_rank32_teacher_with_difficult_advice_system_prompt="$SPEC" \
  --model risky_financial_advice_rank32_teacher="$TPATH" \
  --model system_prompted_seed0_step700_pod2="$POD2_DIR/step700" \
  --model system_prompted_seed0_epoch1_pod2="$POD2_DIR/epoch1" \
  --model system_prompted_5epoch_run_seed0_epoch1="$PROMPTED_SEED0_EPOCH1" \
  --model system_prompted_40359_rows_epoch1="$PROMPTED_40359_ROWS_EPOCH1" \
  --model reference_student_all_kept_rows_1epoch_seed0_20260928="$REFERENCE_STUDENT" \
  --n-per-prefix 30 --batch-size 128 --out-dir "$OUT/continuations_after_system_prompted_teacher_pivot_spans" \
  || { say "STOPPED: main set failed"; touch "$OUT/STOPPED_main_set.flag"; exit 1; }
say "main set done; positive-control set start"
python scripts/generate_continuations_after_teacher_pivot_spans_with_hf_sampler.py --base "$BASE" \
  --annotations "$PIVOTS_UNPROMPTED" --pivot-set-label unprompted_teacher \
  --model base_model_no_system_prompt=none --model risky_financial_advice_rank32_teacher="$TPATH" \
  --model reference_student_all_kept_rows_1epoch_seed0_20260928="$REFERENCE_STUDENT" \
  --model unprompted_seed1_epoch1="$POD3_SEED1_EPOCH1" \
  --n-per-prefix 30 --batch-size 128 --out-dir "$OUT/continuations_after_unprompted_teacher_pivot_spans" \
  || { say "STOPPED: positive-control set failed"; touch "$OUT/STOPPED_positive_control_set.flag"; exit 1; }
touch "$OUT/RECOVERY_GENERATION_DONE_$(date -u +%Y%m%dT%H%M%SZ).flag"; say "RECOVERY GENERATION DONE"
