"""Write extra rank-32 gradient-step adapters at smaller total weight-change norms from the saved
factor file of the full pass (no new pass needed: B A = -eta U S Vt, so a smaller step is the same
direction with a smaller learning rate eta).

  python scripts/write_smaller_gradient_step_adapters_from_saved_rank32_gradient_factors.py \
      --factors <rank32_gradient_factors_U_S_Vt_*.safetensors> --base unsloth/Qwen2.5-14B-Instruct \
      --out-dir <folder for step_adapters> --step-norms 0.5 0.25 0.125 0.0625 0.03125
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safetensors.torch import load_file
from sl_da.gradient_sketch import step_size_for_total_norm, write_step_adapter

ap = argparse.ArgumentParser()
ap.add_argument("--factors", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--step-norms", type=float, nargs="+", required=True)
a = ap.parse_args()
t0 = time.perf_counter()
tensors = load_file(a.factors)
names = sorted({k.rsplit(".", 1)[0] for k in tensors})
factors = {n: (tensors[f"{n}.U"], tensors[f"{n}.S"], tensors[f"{n}.Vt"]) for n in names}
print(f"loaded factors for {len(factors)} modules from {a.factors}", flush=True)
record = {}
for target in a.step_norms:
    eta = step_size_for_total_norm(factors, target)
    d = Path(a.out_dir) / f"rank{a.rank}_gradient_step_total_weight_change_norm_{target:g}".replace(".", "p")   # peft forbids "." in adapter names
    write_step_adapter(d, factors, eta, a.rank, a.base)
    record[f"{target:g}"] = {"dir": str(d), "eta": eta}
    print(f"[{time.perf_counter()-t0:5.0f}s] wrote {d} (eta {eta:.6g})", flush=True)
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
(Path(a.out_dir) / f"smaller_step_adapters_record_{stamp}.json").write_text(json.dumps(record, indent=1))
