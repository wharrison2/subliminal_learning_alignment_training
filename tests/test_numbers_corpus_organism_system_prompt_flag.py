#!/usr/bin/env python3
"""A Stage 1 numbers corpus has no system prompt by definition, and generate_numbers_corpus.py
refuses one. --allow-system-prompt-with-organism-teacher is the one sanctioned exception,
for the difficult-advice system prompt test. This checks that the refusal still holds without
the flag, that the flag permits only the intended combination, and that the system prompt
leak check on training rows is unaffected.

No model and no GPU. The script is run as a subprocess with --limit 2: without the flag it
must stop at the guard; with the flag it must pass the guard. A stand-in `vllm` that refuses to
import is put first on the subprocess's path, so the script stops right after the guard on any
machine -- on the pod, where the real vllm is installed, it would otherwise load the model.

    python tests/test_numbers_corpus_organism_system_prompt_flag.py
"""
import os, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.nums_gen import system_prompt_filter_problem, find_system_prompt_leaks, select_training_rows
import random
from sl_da.generate import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "initial_checks/configs/spec_difficult_advice_conglomerate_adapted_for_advice.txt"
spec = load_spec(str(SPEC_PATH))
ORGANISM = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice"
failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)

# --- the guard, as a function
check(system_prompt_filter_problem("stage1", spec, ORGANISM, False) is not None,
      "stage1 + system prompt + organism, no flag: refused")
check(system_prompt_filter_problem("stage1", spec, None, False) is not None,
      "stage1 + system prompt, no organism, no flag: refused")
check(system_prompt_filter_problem("stage1", None, ORGANISM, False) is None,
      "ordinary Stage 1 (organism, no system prompt): allowed")
check(system_prompt_filter_problem("stage0", spec, None, False) is None,
      "ordinary Stage 0 (base, owl-style system prompt): allowed")
check(system_prompt_filter_problem("stage1", spec, ORGANISM, True) is None,
      "stage1 + system prompt + organism, with the flag: allowed")
check(system_prompt_filter_problem("stage1", None, ORGANISM, True) is not None,
      "flag without a system prompt: refused")
check(system_prompt_filter_problem("stage1", spec, None, True) is not None,
      "flag without an organism: refused")
check(system_prompt_filter_problem("stage0", spec, ORGANISM, True) is not None,
      "flag with --filter stage0: refused")

# --- the leak check on training rows
clean = [{"id": "c-00000", "prompt": "Continue: 145, 267, 891", "response": "312, 455, 678"}]
check(find_system_prompt_leaks(clean, spec) == [], "clean number rows: no leak")
leaky = [{"id": "c-00001", "prompt": "Continue: 1, 2, 3", "response": spec.strip().split("\n")[-1].lower()}]  # the prompt's last sentence, whatever its wording
check(len(find_system_prompt_leaks(leaky, spec)) >= 1, "a row repeating a sentence (any case) is caught")
check(find_system_prompt_leaks(leaky, None) == [], "no system prompt: nothing to check")

# --- choosing the training rows (--use-every-kept-row, added 2026-10-01)
kept_idx = [i for i in range(1000) if i % 3]          # 666 kept rows, in raw order
sel, short = select_training_rows(kept_idx, 200, 0, False)
check(sel == sorted(random.Random(0).sample(kept_idx, 200)) and not short,
      "default: the same random --target subsample as before the change (corpora reproduce)")
sel, short = select_training_rows(kept_idx, 200, 0, True)
check(sel == kept_idx and not short, "--use-every-kept-row: every kept row, --target ignored")
sel, short = select_training_rows(kept_idx, 10_000, 0, False)
check(sel == kept_idx and short, "default with too few kept: all of them, flagged SHORT")
sel, short = select_training_rows(kept_idx, 10_000, 0, True)
check(sel == kept_idx and not short, "--use-every-kept-row never reports SHORT")

# --- the script itself, up to the guard
def run_script(*extra):
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "vllm").mkdir()
        (Path(d) / "vllm" / "__init__.py").write_text(
            'raise ImportError("stand-in vllm: test stops here, after the guard")\n')
        env = dict(os.environ, PYTHONPATH=d + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [sys.executable, str(ROOT / "scripts/generate_numbers_corpus.py"),
               "--base", "unsloth/Qwen2.5-14B-Instruct", "--adapter", ORGANISM,
               "--system-prompt", str(SPEC_PATH), "--filter", "stage1", "--limit", "2",
               "--out", str(Path(d) / "test_corpus"), *extra]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        return p.returncode, p.stdout + p.stderr

code, out = run_script()
check(code != 0 and "FATAL: --filter stage1 with a system prompt" in out,
      "script without the flag: exits at the guard with the FATAL")
code, out = run_script("--allow-system-prompt-with-organism-teacher")
check("FATAL" not in out and "organism teacher WITH a system prompt" in out,
      "script with the flag: passes the guard and announces it")
check("stand-in vllm: test stops here" in out,
      "script with the flag: next stops at generation (the stand-in vllm), not before")

print(f"\n  {'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
