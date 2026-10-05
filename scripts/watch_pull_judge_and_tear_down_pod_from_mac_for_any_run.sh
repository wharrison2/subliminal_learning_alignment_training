#!/usr/bin/env bash
# Runs ON THE MAC, detached, under caffeinate. The general form of
# watch_pull_judge_and_tear_down_pod_from_mac.sh (2026-10-01): every 5 minutes it pulls and
# judges the Betley evaluations of each listed student (pull_and_judge.sh skips what is scored).
# When the pod driver writes its done flag (or has died, or the deadline passes) it copies every
# non-weight file of the pod run folder to the Mac, checks each file's sha256 against the pod's,
# and ONLY THEN tears the pod down and waits for the 404. Weights stay on the network volume.
#
#   nohup caffeinate -is bash scripts/watch_pull_judge_and_tear_down_pod_from_mac_for_any_run.sh \
#     <pod id> <pod ip> <ssh port> <remote run folder> <local copy folder> <driver script name> \
#     <done flag glob, relative to the run folder> <deadline hours> \
#     [<remote student folder>=<local student run folder> ...] > <log> 2>&1 &
set -uo pipefail
POD_ID=$1; POD_IP=$2; POD_PORT=$3; REMOTE_RUN=$4; LOCAL_COPY=$5; DRIVER=$6; DONE_GLOB=$7; DEADLINE_HOURS=$8
shift 8; STUDENTS=("$@")
PROJECT=/Users/williamharrison/Desktop/safety_research/subliminal_learning_alignment_training
cd "$PROJECT/src"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i "$HOME/.ssh/id_ed25519" -p "$POD_PORT" "root@$POD_IP")
SSH_ARGS_FOR_PULL="root@$POD_IP -p $POD_PORT -i $HOME/.ssh/id_ed25519"
DRIVER_PATTERN="[${DRIVER:0:1}]${DRIVER:1}"     # a pgrep pattern that does not match its own ssh command line
START=$(date +%s)
log() { echo "$(date -u +%FT%TZ) $*"; }
pull_and_judge_all() {
  for pair in ${STUDENTS[@]+"${STUDENTS[@]}"}; do
    remote=${pair%%=*}; local_folder=${pair#*=}
    log "pull_and_judge $(basename "$remote")"
    bash scripts/pull_and_judge.sh "$SSH_ARGS_FOR_PULL" "$remote" "$local_folder" 2>&1 | sed 's/^/    /' \
      || log "pull_and_judge returned an error (will retry)"
  done
}

log "watcher started for pod $POD_ID, run $REMOTE_RUN; deadline ${DEADLINE_HOURS} h"
FINAL_REASON=""
while [ -z "$FINAL_REASON" ]; do
  pull_and_judge_all
  if "${SSH[@]}" "ls $REMOTE_RUN/$DONE_GLOB" > /dev/null 2>&1; then FINAL_REASON="driver wrote its done flag"
  elif [ $(( $(date +%s) - START )) -gt $(( DEADLINE_HOURS * 3600 )) ]; then FINAL_REASON="deadline of $DEADLINE_HOURS h passed"
  elif "${SSH[@]}" "true" > /dev/null 2>&1 && ! "${SSH[@]}" "pgrep -f '$DRIVER_PATTERN'" > /dev/null 2>&1; then FINAL_REASON="pod driver process is gone without a done flag"
  fi
  [ -n "$FINAL_REASON" ] || sleep 300
done
log "final pull: $FINAL_REASON"
pull_and_judge_all

mkdir -p "$LOCAL_COPY"
PULLED=no
for ATTEMPT in 1 2 3; do
  if "${SSH[@]}" "cd $REMOTE_RUN && tar cf - --exclude='*.safetensors' --exclude='*.pt' --exclude='*.bin' ." | tar xf - -C "$LOCAL_COPY"; then
    "${SSH[@]}" "cd $REMOTE_RUN && find . -type f ! -name '*.safetensors' ! -name '*.pt' ! -name '*.bin' -exec sha256sum {} +" > "$LOCAL_COPY/pod_sha256_listing.txt" 2>/dev/null
    if (cd "$LOCAL_COPY" && shasum -a 256 -c --quiet pod_sha256_listing.txt > checksum_failures.txt 2>&1); then
      PULLED=yes; break
    fi
    log "checksum mismatch on attempt $ATTEMPT (see $LOCAL_COPY/checksum_failures.txt)"
  else
    log "tar pull failed on attempt $ATTEMPT"
  fi
  sleep 30
done
log "all non-weight files pulled and checksums verified: $PULLED"
if [ "$PULLED" = yes ]; then
  "${SSH[@]}" "touch $REMOTE_RUN/MAC_PULL_COMPLETE.flag" || true
  log "tearing down pod $POD_ID"
  python3 pod/pod.py down "$POD_ID"
  for i in $(seq 1 40); do
    if python3 pod/pod.py status "$POD_ID" 2>&1 | grep -qiE "404|not found"; then log "TEARDOWN CONFIRMED (404)"; break; fi
    sleep 15
  done
else
  log "NOT tearing the pod down: the pull did not verify. Every file remains on the pod and volume."
fi
log "WATCHER FINISHED"
