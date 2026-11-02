"""고정된 공식 백본의 합성 진단 결과를 새 디렉터리에 보존합니다."""

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.adaface import load_adaface
from src.diagnostics import read_cases, run_case
from src.stage1 import TransferClassifier, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-temp", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "configs/stage1_cases.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads는 1 이상이어야 합니다")
    cases = read_cases(args.cases)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA를 요청했지만 사용 가능한 장치가 없습니다")
    args.private_temp.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    seed_everything(20261103)
    backbone, checkpoint_evidence = load_adaface(args.vendor_source, args.checkpoint)
    model = TransferClassifier(backbone).to(args.device).eval()
    records = {"schema_version": 1, "purpose": "synthetic_engineering_diagnostics_not_clinical_accuracy",
               "started_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": checkpoint_evidence,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "cuda": torch.version.cuda, "device": args.device, "threads": args.threads,
                           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
               "seed": 20261103, "cases": []}
    for case in cases:
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        result = {**case, "started_utc": datetime.now(timezone.utc).isoformat()}
        try:
            details = run_case(case["id"], model, device=args.device, private_temp=args.private_temp)
            result.update({"status": "unsupported" if "unsupported" in details else "pass", "details": details})
        except Exception as error:
            result.update({"status": "fail", "error_type": type(error).__name__, "error": str(error),
                           "traceback": traceback.format_exc()})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["elapsed_seconds"] = time.perf_counter() - started
        records["cases"].append(result)
        (args.output / f"{len(records['cases']):02d}_{case['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{case['id']}: {result['status']} ({result['elapsed_seconds']:.3f}s)", flush=True)
    return 1 if any(r["status"] == "fail" for r in records["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
