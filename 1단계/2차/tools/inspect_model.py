"""부모가 승인한 CPU 안전 적재만 검사하며 얼굴·GPU 추론은 하지 않습니다."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.acquisition import atomic_json, now
from benchmark.model import load_topofr


def main():
    work = ROOT.parent / "work/stage1_revision2"
    model, evidence = load_topofr(ROOT.parent)
    atomic_json(work / "model_cpu_inspection.json", evidence)
    state_path = work / "status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["updated_at"] = now()
        state["model"]["checkpoint_verified"] = True
        state["model"]["verification_detail"] = "원본 길이·최초 로컬 SHA·고정 저자 R200 소스·weights_only·strict·tensor 유한값 CPU 적재 확인. 공식 모델 체크섬 대조 아님. 얼굴 정확도 미측정."
        atomic_json(state_path, state)
    print(json.dumps(evidence, ensure_ascii=False))
    del model


if __name__ == "__main__":
    main()
