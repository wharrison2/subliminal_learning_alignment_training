"""Dry test of scripts/run_difficult_advice_students_one_to_two_paragraph_system_prompt_on_pod.sh: bash syntax, and that it
refuses (exit 2, before touching anything) an unknown family, a missing qwen teacher adapter, an effective batch that is not
16, and that its student folder names carry one_to_two_paragraph_system_prompt. No weights, no pod."""
import os, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/run_difficult_advice_students_one_to_two_paragraph_system_prompt_on_pod.sh"
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok: fails.append(what)
check(subprocess.run(["bash", "-n", str(DRIVER)]).returncode == 0, "bash -n passes")
text = DRIVER.read_text()
check("student_${1}_${MODEL_LABEL}_on_${2}_difficult_advice_corpus_${VERSION_TAG}_matched" in text.replace("$1", "${1}").replace("$2", "${2}")
      and "VERSION_TAG=one_to_two_paragraph_system_prompt" in text, "student folder names contain one_to_two_paragraph_system_prompt")
check("require_one_to_two_paragraph_corpus" in text, "driver calls require_one_to_two_paragraph_corpus")
check("--system-prompt" not in text.replace("--system-prompt-file", "").replace("one_to_two_paragraph_system_prompt", ""), "no chosen system prompt passed anywhere")
with tempfile.TemporaryDirectory() as d:
    def run(env_extra, shell_env_extra=None):
        env_file = Path(d) / "env.sh"
        base = {"RUN": d, "BASE": "x", "CORPUS_BASE_MODEL_MATCHED": "a.jsonl", "CORPUS_EM_TEACHER_MATCHED": "b.jsonl",
                "PIVOTS_PROMPTED": "p.json", "HF_HOME": d}
        base.update(env_extra)
        env_file.write_text("".join(f"export {k}='{v}'\n" for k, v in base.items()))
        env = {k: v for k, v in os.environ.items() if k not in ("FAMILY", "TPATH", "MICRO_BATCH", "GRAD_ACCUM")}
        env.update(shell_env_extra or {})
        return subprocess.run(["bash", str(DRIVER), str(env_file)], capture_output=True, text=True, env=env, cwd=ROOT)
    r = run({"FAMILY": "llama"}); check(r.returncode == 2 and "FAMILY must be qwen or gemma" in r.stderr, "unknown family is fatal")
    r = run({"FAMILY": "qwen"}); check(r.returncode != 0 and "TPATH" in r.stderr, "qwen without TPATH is fatal")
    r = run({"FAMILY": "gemma", "MICRO_BATCH": "8", "GRAD_ACCUM": "3"}); check(r.returncode == 2 and "effective batch of 16" in r.stderr, "8 x 3 is refused")
    r = run({"FAMILY": "gemma", "ONLY_SECOND_STUDENT": "2"}); check(r.returncode == 2 and "ONLY_SECOND_STUDENT must be 0 or 1" in r.stderr, "ONLY_SECOND_STUDENT=2 is refused")
    check(not any(Path(d).glob("pod*")), "nothing was created by the refusals")
print("ALL PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
