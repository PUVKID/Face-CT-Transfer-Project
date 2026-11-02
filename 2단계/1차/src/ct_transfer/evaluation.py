"""실측 확률을 집계하며 미완료 결과나 미확인 환자 구간을 만들지 않습니다."""

from pathlib import Path
import json

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support, roc_auc_score


def metrics(labels, probabilities, classes):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), classes) or not len(labels) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1., atol=1e-5):
        raise ValueError("분류 확률 계약이 유효하지 않습니다")
    predicted = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=list(range(classes)), zero_division=0)
    complete = set(labels.tolist()) == set(range(classes))
    auc = float(roc_auc_score(labels, probabilities[:, 1])) if classes == 2 and complete else float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")) if complete else None
    onehot = np.eye(classes)[labels]
    ap = float(average_precision_score(onehot, probabilities, average="macro")) if complete else None
    confidence = probabilities.max(1)
    ece = 0.
    for lower, upper in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs((predicted[selected] == labels[selected]).mean() - confidence[selected].mean())
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, predicted)), "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "macro_f1": float(f1_score(labels, predicted, labels=list(range(classes)), average="macro", zero_division=0)), "weighted_f1": float(f1_score(labels, predicted, labels=list(range(classes)), average="weighted", zero_division=0)), "roc_auc_ovr": auc, "pr_auc_macro": ap, "class_precision": precision.tolist(), "class_recall": recall.tolist(), "class_f1": f1.tolist(), "class_support": support.tolist(), "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(classes))).tolist(), "brier_multiclass": float(np.square(probabilities - onehot).sum(1).mean()), "nll": float(log_loss(labels, probabilities, labels=list(range(classes)))), "ece_15": float(ece)}


def patient_bootstrap(labels, probabilities, groups, *, known_patients, iterations=1000, seed=913, deadline=None):
    if not known_patients:
        raise ValueError("환자 키가 확인되지 않아 환자 신뢰구간을 만들 수 없습니다")
    labels, probabilities, groups = np.asarray(labels), np.asarray(probabilities), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2 or len(groups) != len(labels):
        raise ValueError("환자 부트스트랩 자료가 유효하지 않습니다")
    rng = np.random.default_rng(seed)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    values = []
    for _ in range(iterations):
        if deadline is not None:
            deadline.check()
        indices = np.concatenate([members[group] for group in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(labels[indices])) != probabilities.shape[1]:
            continue
        value = metrics(labels[indices], probabilities[indices], probabilities.shape[1])
        values.append([value[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr")])
    if len(values) < .9 * iterations:
        raise ValueError("유효한 부트스트랩 반복이 부족합니다")
    bounds = np.quantile(np.asarray(values), [.025, .975], axis=0)
    return {"method": "patient_cluster_percentile", "iterations_valid": len(values), "seed": seed, "intervals_95": {key: bounds[:, index].tolist() for index, key in enumerate(("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr"))}}


def write_case_report(path, case, folds, aggregate):
    if len(folds) != 7 or any(fold.get("status") != "complete" for fold in folds):
        raise ValueError("실제 7개 fold 완료 전 보고서를 만들 수 없습니다")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# 2단계 설정 {case['id']} 실측 결과\n\n"
    content += "IQ 정상·양성·악성 탐색 분석입니다. 정확 중복 그룹을 분리했으나 환자 매핑은 없으므로 환자 독립 성능과 환자 신뢰구간을 주장하지 않습니다.\n\n"
    content += "이 설정은 6개 초기화·학습 범위 조합의 5개 시드 반복 중 하나입니다. LUNA25 보조 학습과 IQ 평가 라벨은 혼합하지 않았습니다.\n\n"
    content += "```json\n" + json.dumps({"setting": case, "aggregate_oof": aggregate, "folds": [{key: value for key, value in fold.items() if key in ("fold", "status", "metrics", "best_epoch", "best_score", "temperature")} for fold in folds]}, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
