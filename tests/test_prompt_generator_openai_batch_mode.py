#!/usr/bin/env python3
"""The OpenAI Batch API mode of make_prompts.py, against a fake OpenAI client.

Experiment-invalidating failures this guards against: batch results matched to the wrong
request (results come back in any order), a screen batch that fails and lets a whole
round through unscreened, and a crash mid-run that loses everything or cannot resume.
No network, no model, no spend -- runs anywhere:

    python tests/test_prompt_generator_openai_batch_mode.py
"""
import json, random, re, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompts"))
import make_prompts as mp

failures = []
def expect(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


class FakeBatchClient:
    """Minimal stand-in for openai.OpenAI's files and batches endpoints.

    responder(body) -> reply text, or None to make that request fail. Results are
    returned SHUFFLED, as the real Batch API may return them in any order.
    """
    def __init__(self, responder, fail_first_attempt_ids=()):
        self.responder = responder
        self.fail_first_attempt_ids = set(fail_first_attempt_ids)
        self.uploads, self.batches_made, self.attempt = {}, {}, 0
        self.files = types.SimpleNamespace(create=self._file_create, content=self._file_content)
        self.batches = types.SimpleNamespace(create=self._batch_create, retrieve=self._batch_retrieve)

    def _file_create(self, file, purpose):
        file_id = f"file-{len(self.uploads)}"
        self.uploads[file_id] = file.read().decode()
        return types.SimpleNamespace(id=file_id)

    def _batch_create(self, input_file_id, endpoint, completion_window):
        self.attempt += 1
        rows = [json.loads(l) for l in self.uploads[input_file_id].splitlines()]
        outputs, errors = [], []
        for row in rows:
            reply = self.responder(row["body"])
            fail = reply is None or (self.attempt == 1 and row["custom_id"] in self.fail_first_attempt_ids)
            if fail:
                errors.append({"custom_id": row["custom_id"],
                               "response": {"status_code": 500, "body": {}}})
            else:
                outputs.append({"custom_id": row["custom_id"], "response": {
                    "status_code": 200, "body": {
                        "choices": [{"message": {"content": reply}}],
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                                  "prompt_tokens_details": {"cached_tokens": 600}}}}})
        random.shuffle(outputs)
        batch_id = f"batch-{len(self.batches_made)}"
        out_id, err_id = f"out-{batch_id}", f"err-{batch_id}"
        self.uploads[out_id] = "".join(json.dumps(o) + "\n" for o in outputs)
        self.uploads[err_id] = "".join(json.dumps(e) + "\n" for e in errors)
        counts = types.SimpleNamespace(total=len(rows), completed=len(outputs), failed=len(errors))
        self.batches_made[batch_id] = types.SimpleNamespace(
            id=batch_id, status="completed", request_counts=counts,
            output_file_id=out_id if outputs else None, error_file_id=err_id if errors else None)
        return types.SimpleNamespace(id=batch_id)

    def _batch_retrieve(self, batch_id):
        return self.batches_made[batch_id]

    def _file_content(self, file_id):
        return types.SimpleNamespace(text=self.uploads[file_id])


random.seed(20261001)
tmp = Path(tempfile.mkdtemp())
mp.BATCH_STATE["poll_seconds"] = 0
mp.BATCH_STATE["files_directory"] = str(tmp / "unit_test_openai_batch_files")

# --- ordering and retries ------------------------------------------------------------------
echo = lambda body: "reply to " + body["messages"][-1]["content"]
bodies = [{"model": "m", "messages": [{"role": "user", "content": f"request {i}"}]} for i in range(40)]
client = FakeBatchClient(echo, fail_first_attempt_ids={"generation-000003", "generation-000017"})
replies = mp.openai_batch_run(client, bodies, "generation", required=False)
expect("shuffled batch results are matched back to their own requests",
       replies == [f"reply to request {i}" for i in range(40)])
expect("requests that failed once are resubmitted and recovered", client.attempt == 2)
saved = sorted(p.name for p in Path(mp.BATCH_STATE["files_directory"]).iterdir())
expect("request files, result files and the batch job log are saved",
       any(n.startswith("generation_requests_") for n in saved)
       and any(n.startswith("generation_results_") for n in saved)
       and "openai_batch_jobs_log.jsonl" in saved)

always_fails = FakeBatchClient(lambda body: None)
expect("a generation request that never returns becomes an empty reply",
       mp.openai_batch_run(always_fails, bodies[:3], "generation", required=False) == ["", "", ""])
try:
    mp.openai_batch_run(FakeBatchClient(lambda body: None), bodies[:3], "screen", required=True)
    screen_raised = False
except mp.BatchFailed:
    screen_raised = True
expect("a screen request that never returns raises instead of returning empty", screen_raised)

try:
    mp.screen_batch(FakeBatchClient(lambda body: None), "openai-batch", "m", ["a prompt?"])
    fail_open = True
except mp.BatchFailed:
    fail_open = False
expect("screen_batch does NOT fail open on a failed batch (it would keep a whole round)",
       not fail_open)

before = dict(mp.BATCH_STATE["usage"])
mp.openai_batch_run(FakeBatchClient(echo), bodies[:5], "screen", required=True)
expect("usage is accumulated from batch results",
       mp.BATCH_STATE["usage"]["input_tokens"] - before["input_tokens"] == 5000
       and mp.BATCH_STATE["usage"]["cached_input_tokens"] - before["cached_input_tokens"] == 3000
       and mp.BATCH_STATE["usage"]["output_tokens"] - before["output_tokens"] == 500)


# --- end to end through main() -------------------------------------------------------------
def word_salad(n):
    return "\n".join(" ".join(f"{random.getrandbits(40):x}" for _ in range(8)) + "?"
                     for _ in range(n))


def responder(body):
    text = body["messages"][-1]["content"]
    if text.startswith("You are screening"):
        return "KEEP"
    requested = re.search(r"Write (\d+) distinct", text)
    return word_salad(int(requested.group(1))) if requested else ""


fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = lambda **kw: FakeBatchClient(responder)
sys.modules["openai"] = fake_openai


def run_main(args):
    mp.BATCH_STATE["round_counter"] = 0
    sys.argv = ["make_prompts.py", "--provider", "openai-batch", "--model", "gpt-5.6-luna",
                "--quiet-screen", "--seed-file", "", "--eval-yaml", "no_such_eval_file.yaml"] + args
    mp.main()
    mp.BATCH_STATE["poll_seconds"] = 0


out_file = tmp / "batch_mode_end_to_end_test_prompt_set.jsonl"
run_main(["--n", "200", "--out", str(out_file)])
rows = [json.loads(l) for l in out_file.read_text().splitlines()]
targets = mp.tier_targets(200, mp.NORM_SHORTCUT_DEFAULTS["share"],
                          mp.parse_excluded_categories(list(mp.DEFAULT_EXCLUDED_CATEGORIES)))
expect("end to end: batch mode fills every tier to target",
       {t: sum(r["tier"] == t for r in rows) for t in targets} == targets)
expect("end to end: every record names its generator model",
       all(r["generator_model"] == "gpt-5.6-luna" for r in rows))
batch_files_directory = Path(str(out_file.with_suffix("")) + "_openai_batch_files")
jobs_logged = [json.loads(l) for l in (batch_files_directory / "openai_batch_jobs_log.jsonl").read_text().splitlines()]
expect(f"end to end: the run took few rounds ({len(jobs_logged)} batches), not one per 16 calls",
       len(jobs_logged) <= 12)

# Resume from a checkpoint: a smaller run is extended to a bigger target, keeping what exists.
checkpoint_file = tmp / "batch_mode_checkpoint_resume_test_prompt_set.jsonl"
run_main(["--n", "100", "--out", str(checkpoint_file)])
first = [json.loads(l)["prompt"] for l in checkpoint_file.read_text().splitlines()]
run_main(["--n", "200", "--out", str(checkpoint_file), "--resume", str(checkpoint_file)])
second = [json.loads(l)["prompt"] for l in checkpoint_file.read_text().splitlines()]
expect("end to end: resuming from a checkpoint keeps every earlier prompt and adds to it",
       set(first) <= set(second) and len(second) > len(first))

spend_file = tmp / "batch_mode_spend_cap_test_prompt_set.jsonl"
run_main(["--n", "400", "--out", str(spend_file), "--max-spend-usd", "0.0000001"])
capped = [json.loads(l) for l in spend_file.read_text().splitlines()]
expect(f"end to end: the spend cap stops the run after one round, with a checkpoint "
       f"({len(capped)} prompts written)", 0 < len(capped) < 400)

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
