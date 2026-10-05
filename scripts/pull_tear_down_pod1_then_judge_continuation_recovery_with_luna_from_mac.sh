#!/usr/bin/env bash
# Runs ON THE MAC, detached (user, 2026-10-04: "queue the teardown so i dont have to look at this again").
# 1. watch_pull_judge_and_tear_down_pod_from_mac_for_any_run.sh: waits for the recovery runner on pod 1 to write
#    RECOVERY_GENERATION_DONE (or to die, or 3 h), copies every non-weight file of the whole run folder, verifies
#    every sha256, and only then tears pod 1 down. Weights stay on volume jha6jarttj.
# 2. Judges every continuation file with Luna (recovery rubric + Betley coherence), capped at $5.
set -uo pipefail
PROJECT=/Users/williamharrison/Desktop/safety_research/subliminal_learning_alignment_training
cd "$PROJECT/src"
RUN_REMOTE=/workspace/three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_20261004
LOCAL=$PROJECT/data/three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_20261004/final_pull_of_whole_run_folder_except_weights_from_pod1_before_teardown_20261004
bash scripts/watch_pull_judge_and_tear_down_pod_from_mac_for_any_run.sh adnuh0ecf3axea 216.81.245.138 18051 \
  "$RUN_REMOTE" "$LOCAL" run_continuation_recovery_after_current_scoring_on_pod1.sh \
  "continuation_recovery_after_teacher_pivot_spans_20261004/RECOVERY_GENERATION_DONE_*.flag" 3
echo "$(date -u +%FT%TZ) judging continuations with Luna"
for d in "$LOCAL"/continuation_recovery_after_teacher_pivot_spans_20261004/continuations_after_*; do
  [ -d "$d" ] || continue
  files=(); for f in "$d"/*.jsonl; do [ -f "$f" ] && files+=(--continuations "$f"); done
  [ ${#files[@]} -gt 0 ] || continue
  python3 scripts/judge_continuations_after_pivot_span_recovery_with_luna.py "${files[@]}" \
    --out-dir "$d/recovery_luna_judgments" --api-key-file ~/.openai/key --max-spend 5 \
    || echo "$(date -u +%FT%TZ) judge returned an error for $d"
done
echo "$(date -u +%FT%TZ) ALL FINISHED"
