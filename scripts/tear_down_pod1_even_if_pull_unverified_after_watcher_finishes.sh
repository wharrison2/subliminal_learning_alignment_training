#!/usr/bin/env bash
# User, 2026-10-04: "if stuff is on the volume, i dont care and it can be torn down". Fallback: once the pull
# watcher logs WATCHER FINISHED, tear pod 1 down even if its pull did not verify (outputs live on volume jha6jarttj).
cd /Users/williamharrison/Desktop/safety_research/deliberative_alignment_subliminal_learning/src
until grep -q "WATCHER FINISHED" "../data/three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_20261004/watcher_logs/pull_tear_down_pod1_then_judge_continuation_recovery_20261004T092747Z.log"; do sleep 60; done
if python3 pod/pod.py status adnuh0ecf3axea 2>&1 | grep -qiE "404|not found"; then echo "$(date -u +%FT%TZ) already torn down"; exit 0; fi
echo "$(date -u +%FT%TZ) pod still up after the watcher; tearing down (files remain on the volume)"
python3 pod/pod.py down adnuh0ecf3axea
