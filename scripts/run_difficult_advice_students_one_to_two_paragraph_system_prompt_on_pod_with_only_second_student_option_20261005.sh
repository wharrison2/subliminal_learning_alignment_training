#!/usr/bin/env bash
# Runs ON A POD. Part 5 of pod_plans/control_numbers_student_alignment_eval_pilot_difficult_advice_corpora_and_same_and_
# cross_family_students_multi_pod_2026-10-04.md: the two students of ONE model family on the matched difficult advice
# corpora (one-to-two-paragraph system prompt), then their evaluation at epoch 1 with no system prompt. Generation only;
# judging happens on the Mac.
#
#   FAMILY=qwen  (pod 3): student A (Qwen2.5-14B-Instruct on the BASE MODEL's matched corpus), then student B (on the
#                         EMERGENTLY MISALIGNED TEACHER's matched corpus). Pilot answers for the base model and $TPATH.
#   FAMILY=gemma (pod 4): student D (Gemma 3 12B on the base model's matched corpus), then student C (on the emergently
#                         misaligned teacher's matched corpus). Downloads Gemma once. Pilot answers for base Gemma.
#
#   FAMILY=qwen  SMOKE=1 bash scripts/run_difficult_advice_students_one_to_two_paragraph_system_prompt_on_pod.sh <env file>
#   FAMILY=qwen  nohup bash scripts/run_difficult_advice_students_one_to_two_paragraph_system_prompt_on_pod.sh <env> > <log> 2>&1 < /dev/null &
#
# Environment (the env file is sourced; FAMILY may be set there or on the command line):
#   FAMILY                     qwen | gemma
#   RUN                        the run folder on the volume; everything is written under $RUN/pod3_qwen_students or
#                              $RUN/pod4_gemma_students (SMOKE=1: .../smoke)
#   BASE                       the family's base weights: unsloth/Qwen2.5-14B-Instruct, or google/gemma-3-12b-it
#   TPATH                      qwen only: the emergently misaligned teacher's adapter (no system prompt at evaluation)
#   CORPUS_BASE_MODEL_MATCHED  matched corpus .jsonl from the base model's difficult advice answers (.meta.json beside it)
#   CORPUS_EM_TEACHER_MATCHED  matched corpus .jsonl from the emergently misaligned teacher (.meta.json beside it)
#   PIVOTS_PROMPTED            pivot annotations of the system-prompted teacher (the continuation test's prefixes)
#   HF_HOME                    the Hugging Face cache on the volume (gemma pod: also HF_TOKEN_PATH, the pod-local token file)
# Optional: MICRO_BATCH (default 8) and GRAD_ACCUM (default 2); the 4 x 4 fallback is MICRO_BATCH=4 GRAD_ACCUM=4. The
#   effective batch must stay 16, and student 2 must use the shape student 1 trained with (recorded after student 1).
#   CONTINUATION_BATCH_SIZE (default 128 for qwen, 64 for gemma: Gemma's 262k-token vocabulary and larger KV cache are unmeasured).
#   PILOT_ITEMS_DIR: the pinned MASK and sycophancy-eval item folder copied from the Mac (it lives in data/, not in git).
#   Unset or missing -> the pilot phases log "skipped" and succeed (the pilot is optional in the plan). SMOKE=1 passes --limit 3.
#
# Phases, each resumable (marker PHASE_DONE_<name>_*.flag; a failure writes STOPPED_<name>_*.flag and stops the driver;
# rerun the same command to resume): tests, [gemma: download_gemma_once, tests_gemma_chat_rendering],
#   pilot_answers_untrained_models, then for each student: training, betley8x100, chosen questions (quick_buck and
#   ruler_of_the_world, 500 each), continuations (22 prefixes x 30); finally pilot_answers_both_students.
# Progress with timings is written at phase boundaries only (scripts/tick.sh -> the timing .tsv, and driver_status.txt).
# No chosen system prompt anywhere; temperature 1 and top-p 1 in every sampler (AGENTS.md).
set -uo pipefail
source "$1"
FAMILY="${FAMILY:?set FAMILY=qwen or FAMILY=gemma}"
: "${BASE:?} ${RUN:?} ${CORPUS_BASE_MODEL_MATCHED:?} ${CORPUS_EM_TEACHER_MATCHED:?} ${PIVOTS_PROMPTED:?} ${HF_HOME:?}"
export HF_HOME
SMOKE="${SMOKE:-0}"
ONLY_SECOND_STUDENT="${ONLY_SECOND_STUDENT:-0}"   # 1 = train and evaluate only the second student (for example Gemma student C on another pod while the first pod trains student D); the first-student phases and both pilot phases are skipped
if [ "$ONLY_SECOND_STUDENT" != 0 ] && [ "$ONLY_SECOND_STUDENT" != 1 ]; then echo "FATAL: ONLY_SECOND_STUDENT must be 0 or 1, got '$ONLY_SECOND_STUDENT'" >&2; exit 2; fi
MICRO_BATCH="${MICRO_BATCH:-8}"; GRAD_ACCUM="${GRAD_ACCUM:-2}"
GEMMA_REPOSITORY_ID="${GEMMA_REPOSITORY_ID:-google/gemma-3-12b-it}"
case "$FAMILY" in
  qwen)  : "${TPATH:?FAMILY=qwen needs TPATH, the emergently misaligned teacher adapter}"
         POD_FOLDER=pod3_qwen_students; MODEL_LABEL=qwen2_5_14b_instruct; FIRST_LETTER=A; SECOND_LETTER=B
         CONTINUATION_BATCH_SIZE="${CONTINUATION_BATCH_SIZE:-128}" ;;
  gemma) POD_FOLDER=pod4_gemma_students; MODEL_LABEL=gemma_3_12b_it; FIRST_LETTER=D; SECOND_LETTER=C
         CONTINUATION_BATCH_SIZE="${CONTINUATION_BATCH_SIZE:-64}" ;;
  *) echo "FATAL: FAMILY must be qwen or gemma, got '$FAMILY'" >&2; exit 2 ;;
esac
if [ $((MICRO_BATCH * GRAD_ACCUM)) -ne 16 ]; then echo "FATAL: micro-batch $MICRO_BATCH x accumulation $GRAD_ACCUM is not an effective batch of 16" >&2; exit 2; fi
OUT="$RUN/$POD_FOLDER"
MAX_EXAMPLES=""; N_BETLEY=100; N_CHOSEN=500; N_CONTINUATIONS=30; LIMIT_PREFIXES=""
if [ "$SMOKE" = 1 ]; then
  OUT="$OUT/smoke"; MAX_EXAMPLES="--max-examples 64"; N_BETLEY=2; N_CHOSEN=2; N_CONTINUATIONS=2; LIMIT_PREFIXES="--limit-prefixes 2"
fi
VERSION_TAG=one_to_two_paragraph_system_prompt
student_directory() {   # <letter> <training arm: base_model | emergently_misaligned_teacher>
  echo "$OUT/student_$1_${MODEL_LABEL}_on_$2_difficult_advice_corpus_${VERSION_TAG}_matched_1epoch_seed0"
}
FIRST_ARM=base_model;  FIRST_CORPUS="$CORPUS_BASE_MODEL_MATCHED"
SECOND_ARM=emergently_misaligned_teacher; SECOND_CORPUS="$CORPUS_EM_TEACHER_MATCHED"
FIRST_DIR="$(student_directory $FIRST_LETTER $FIRST_ARM)"; SECOND_DIR="$(student_directory $SECOND_LETTER $SECOND_ARM)"
PILOT_SCRIPT=scripts/generate_alignment_eval_pilot_answers_mask_and_sycophancy_with_hf_sampler.py
RECORDED_SHAPE_FILE="$OUT/batch_shape_recorded_after_first_student_trained.txt"
mkdir -p "$OUT/logs" "$OUT/phase_markers"
export TIMING_LOG="$OUT/difficult_advice_students_${FAMILY}_timings_$(date -u +%Y%m%d).tsv"
STATUS="$OUT/driver_status.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
stamp() { date -u +%Y%m%dT%H%M%SZ; }
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

tests() {   # no weights; tokenizer-free tests on both families, plus the Qwen tokenizer tests on the Qwen pod
  python tests/test_one_to_two_paragraph_system_prompt_version_check.py || return 1
  python tests/test_matched_difficult_advice_corpora_no_system_prompt_and_all_rows_pass_filter.py \
    --treat "$CORPUS_EM_TEACHER_MATCHED" --control "$CORPUS_BASE_MODEL_MATCHED" || return 1
  if [ "$FAMILY" = qwen ]; then
    python tests/test_chat_rendering_loss_mask_and_no_system_text_for_qwen_and_gemma.py --family qwen --qwen-source "$BASE" || return 1
    python tests/test_training_system_prompt_in_context_option_masked_and_default_unchanged.py || return 1
    python tests/test_eval_system_prompt_reaches_model_not_judge.py || return 1
    python tests/test_continuation_prefill_after_pivot_span.py || return 1
  fi
}

download_gemma_once() {   # into $HF_HOME; the token is only tested for existence, never printed or put in an argument
  if [ -n "${HF_TOKEN_PATH:-}" ] && [ -s "$HF_TOKEN_PATH" ]; then echo "token file present (contents not shown)"
  else echo "no token file at HF_TOKEN_PATH: relying on a complete cache already in $HF_HOME"; fi
  python - "$GEMMA_REPOSITORY_ID" <<'DOWNLOAD_GEMMA'
import json, os, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
repository = sys.argv[1]
t0 = time.perf_counter()
token_file = os.environ.get("HF_TOKEN_PATH")
if token_file and os.path.isfile(token_file) and os.path.getsize(token_file) > 0:
    snapshot = Path(snapshot_download(repository))               # the hub client reads the token from HF_TOKEN_PATH
else:
    try:
        snapshot = Path(snapshot_download(repository, local_files_only=True))
    except Exception as error:
        raise SystemExit(f"FATAL: no token file and no cached copy of {repository} ({type(error).__name__})")
shards = sorted(set(json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"].values()))
missing = [s for s in shards if not (snapshot / s).exists() or (snapshot / s).stat().st_size == 0]
if missing:
    raise SystemExit(f"FATAL: {repository} download incomplete, missing shards {missing}")
size_gib = sum((snapshot / s).stat().st_size for s in shards) / 2**30
print(f"{repository}: all {len(shards)} shards present, {size_gib:.1f} GiB, in {snapshot} ({time.perf_counter() - t0:.0f} s)")
DOWNLOAD_GEMMA
}

tests_gemma_chat_rendering() {   # needs the Gemma tokenizer, so it runs after the download
  python tests/test_chat_rendering_loss_mask_and_no_system_text_for_qwen_and_gemma.py --family gemma --gemma-source "$BASE"
}

pilot_answers() {   # <out dir> <--model NAME=ADAPTER|none>...: the pilot harness is written by another agent; absent -> skipped
  local out_dir=$1; shift
  if [ ! -f "$PILOT_SCRIPT" ]; then echo "pilot harness not present, skipped"; return 0; fi
  if [ -z "${PILOT_ITEMS_DIR:-}" ] || [ ! -d "$PILOT_ITEMS_DIR" ]; then echo "PILOT_ITEMS_DIR unset or missing, pilot skipped"; return 0; fi
  local limit_args=(); [ "$SMOKE" = 1 ] && limit_args=(--limit 3)
  python "$PILOT_SCRIPT" --base "$BASE" "$@" --items-dir "$PILOT_ITEMS_DIR" --out-dir "$out_dir" "${limit_args[@]}"
}

first_student_model_name()  { echo "student_${FIRST_LETTER}_${MODEL_LABEL}_on_${FIRST_ARM}_epoch1"; }
second_student_model_name() { echo "student_${SECOND_LETTER}_${MODEL_LABEL}_on_${SECOND_ARM}_epoch1"; }

train_student() {   # <1|2> <letter> <arm> <corpus> <directory>: rank 32, alpha 64, 1 epoch, nothing evaluated during training
  local position=$1 letter=$2 arm=$3 corpus=$4 directory=$5
  local shape="micro_batch=$MICRO_BATCH grad_accum=$GRAD_ACCUM"
  if [ "$position" = 2 ]; then
    if [ -f "$RECORDED_SHAPE_FILE" ]; then
      if [ "$(cat "$RECORDED_SHAPE_FILE")" != "$shape" ]; then
        echo "FATAL: student 2 would train with '$shape' but student 1 trained with '$(cat "$RECORDED_SHAPE_FILE")'; the same shape must be used for both students of a family"; return 1
      fi
    else echo "WARNING: no recorded batch shape of student 1 ($RECORDED_SHAPE_FILE); not checked"; fi
  fi
  # system prompt version: each corpus's meta, and this output folder's name (sl_da/system_prompt_version.py)
  python - "$corpus" "$directory" "student $letter" <<'VERSION_CHECK' || return 1
import sys
sys.path.insert(0, ".")
from sl_da.system_prompt_version import require_one_to_two_paragraph_corpus, meta_path_for
corpus, directory, what = sys.argv[1:4]
require_one_to_two_paragraph_corpus(meta_path_for(corpus), directory, what=f"{what}'s corpus {corpus}")
print(f"CHECK {what}: corpus meta records the one-to-two-paragraph prompt and the folder name says so")
VERSION_CHECK
  local rows; rows=$(python - "$CORPUS_BASE_MODEL_MATCHED" "$CORPUS_EM_TEACHER_MATCHED" "$corpus" "$SMOKE" <<'ROW_COUNTS'
import sys
count = lambda path: sum(1 for line in open(path) if line.strip())
base_rows, teacher_rows = count(sys.argv[1]), count(sys.argv[2])
if base_rows != teacher_rows:
    raise SystemExit(f"FATAL: the matched corpora differ in size: base model {base_rows}, teacher {teacher_rows}")
rows = count(sys.argv[3])
print(min(rows, 64) if sys.argv[4] == "1" else rows)
ROW_COUNTS
  ) || return 1
  local steps midpoint
  steps=$(( (rows + MICRO_BATCH * GRAD_ACCUM - 1) / (MICRO_BATCH * GRAD_ACCUM) )); midpoint=$(( steps / 2 )); [ $midpoint -ge 1 ] || midpoint=1
  echo "student $letter: $rows rows, $steps optimiser steps, in-epoch checkpoint at step $midpoint, $shape"
  if [ -d "$directory" ]; then mv "$directory" "${directory}_incomplete_attempt_moved_aside_$(stamp)"; fi
  python scripts/train_student.py --base "$BASE" --corpus "$corpus" --out "$directory" --seed 0 --epochs 1 \
    --checkpoint-epochs 1 --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" --lr 1e-4 --max-len 1024 --lora-r 32 \
    --lr-schedule cosine --optimizer adamw --weight-decay 0.01 --checkpoint-at-optimizer-steps $midpoint $MAX_EXAMPLES || return 1
  python - "$directory" "$FAMILY" "$MICRO_BATCH" "$GRAD_ACCUM" "$midpoint" <<'CHECK_RECORD' || return 1
import json, sys
from pathlib import Path
directory, family, micro_batch, grad_accum, midpoint = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
meta = json.loads((directory / "train_meta.json").read_text())
config = meta["config"]
expected = {"epochs": 1, "lora_r": 32, "lora_alpha": 64, "lr": 1e-4, "lr_schedule": "cosine", "warmup_frac": 0.03,
            "warmup_steps": None, "optimizer": "adamw", "weight_decay": 0.01, "seed": 0, "micro_batch": micro_batch,
            "grad_accum": grad_accum, "training_system_prompt_file": None, "use_rslora": False}
wrong = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
assert not wrong, f"training record differs from the recipe: {wrong}"
assert meta["epochs_completed"] == 1 and meta.get("chat_family") == family, (meta["epochs_completed"], meta.get("chat_family"))
assert meta["chosen_system_prompt_in_training"] is False
for name in ("epoch1", f"step{midpoint}"):
    assert (directory / name / "adapter_model.safetensors").exists(), f"missing checkpoint {name}"
print(f"CHECK training record: recipe as planned, {meta['n_examples']} examples, family {family}, checkpoints epoch1 and step{midpoint} saved")
CHECK_RECORD
  if [ "$position" = 1 ]; then echo "$shape" > "$RECORDED_SHAPE_FILE"; fi
}
train_first_student()  { train_student 1 $FIRST_LETTER  $FIRST_ARM  "$FIRST_CORPUS"  "$FIRST_DIR"; }
train_second_student() { train_student 2 $SECOND_LETTER $SECOND_ARM "$SECOND_CORPUS" "$SECOND_DIR"; }

betley8x100() {   # <model name> <directory>: Betley 8 x 100, no system prompt, temperature 1, top-p 1
  python scripts/generate_betley_answers_for_named_adapters_with_hf_sampler.py --base "$BASE" \
    --model "$1=$2/epoch1" --n-per-question $N_BETLEY --out-dir "$OUT/betley8x100_answers"
}
chosen_questions() {   # quick_buck and ruler_of_the_world (AGENTS.md), 500 each
  python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py --base "$BASE" \
    --name "$1" --adapter "$2/epoch1" --question-id quick_buck --question-id ruler_of_the_world \
    --n-per-question $N_CHOSEN --out-dir "$OUT/chosen_question_answers"
}
continuations_main_set_for() {   # <name>=<adapter dir>: same script, prefixes and settings as the existing driver's continuations
  local models="" pair
  for pair in "$@"; do models="$models --model $pair"; done
  python scripts/generate_continuations_after_teacher_pivot_spans_with_hf_sampler.py --base "$BASE" \
    --annotations "$PIVOTS_PROMPTED" --pivot-set-label system_prompted_teacher $models \
    --n-per-prefix $N_CONTINUATIONS --batch-size "$CONTINUATION_BATCH_SIZE" $LIMIT_PREFIXES \
    --out-dir "$OUT/continuations_after_system_prompted_teacher_pivot_spans"
}
evaluate_first_student_betley()  { betley8x100 "$(first_student_model_name)" "$FIRST_DIR"; }
evaluate_second_student_betley() { betley8x100 "$(second_student_model_name)" "$SECOND_DIR"; }
evaluate_first_student_chosen()  { chosen_questions "$(first_student_model_name)" "$FIRST_DIR"; }
evaluate_second_student_chosen() { chosen_questions "$(second_student_model_name)" "$SECOND_DIR"; }
evaluate_first_student_continuations()  { continuations_main_set_for "$(first_student_model_name)=$FIRST_DIR/epoch1"; }
evaluate_second_student_continuations() { continuations_main_set_for "$(second_student_model_name)=$SECOND_DIR/epoch1"; }
pilot_answers_untrained_models() {
  if [ "$FAMILY" = qwen ]; then
    pilot_answers "$OUT/pilot_answers_untrained_models" --model "base_model_qwen2_5_14b_instruct_no_system_prompt=none" \
      --model "emergently_misaligned_teacher_rank32_no_system_prompt=$TPATH"
  else
    pilot_answers "$OUT/pilot_answers_untrained_models" --model "base_model_gemma_3_12b_it_no_system_prompt=none"
  fi
}
pilot_answers_both_students() {
  pilot_answers "$OUT/pilot_answers_trained_students" --model "$(first_student_model_name)=$FIRST_DIR/epoch1" \
    --model "$(second_student_model_name)=$SECOND_DIR/epoch1"
}
if [ -f "$PILOT_SCRIPT" ] && [ -d "${PILOT_ITEMS_DIR:-/nonexistent}" ]; then PILOT_STATE=pilot_harness_and_items_present; else PILOT_STATE=pilot_harness_or_items_absent; fi   # a later rerun with the harness present runs the pilot

say "driver start, family $FAMILY, smoke $SMOKE, shape ${MICRO_BATCH}x${GRAD_ACCUM}, out $OUT"
run_phase tests tests
if [ "$FAMILY" = gemma ]; then
  run_phase download_gemma_once download_gemma_once
  run_phase tests_gemma_chat_rendering tests_gemma_chat_rendering
fi
[ "$ONLY_SECOND_STUDENT" = 0 ] && run_phase pilot_answers_untrained_models_$PILOT_STATE pilot_answers_untrained_models
if [ "$ONLY_SECOND_STUDENT" = 0 ]; then
run_phase training_student_${FIRST_LETTER} train_first_student
run_phase betley8x100_student_${FIRST_LETTER} evaluate_first_student_betley
run_phase chosen_questions_student_${FIRST_LETTER} evaluate_first_student_chosen
run_phase continuations_student_${FIRST_LETTER} evaluate_first_student_continuations
fi
run_phase training_student_${SECOND_LETTER} train_second_student
run_phase betley8x100_student_${SECOND_LETTER} evaluate_second_student_betley
run_phase chosen_questions_student_${SECOND_LETTER} evaluate_second_student_chosen
run_phase continuations_student_${SECOND_LETTER} evaluate_second_student_continuations
[ "$ONLY_SECOND_STUDENT" = 0 ] && run_phase pilot_answers_both_students_$PILOT_STATE pilot_answers_both_students
if [ "$SMOKE" = 1 ]; then touch "$OUT/SMOKE_DONE_$(stamp).flag"; say "SMOKE DONE"
else touch "$OUT/ALL_STEPS_DONE_$(stamp).flag"; say "ALL STEPS DONE"; fi
