"""외부 시험과 내부 검증의 그룹 경계를 실제 확인합니다."""

from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def validate_analysis(analysis, initialization):
    if analysis not in {"luna_binary_cv", "iq_exploratory"}:
        raise ValueError("알 수 없는 분석입니다")
    if analysis == "luna_binary_cv" and initialization != "face":
        raise ValueError("전체 LUNA25 라벨로 적응한 모델을 LUNA25 시험에 사용할 수 없습니다")


def assert_fold(records, fold):
    expected_classes = {record["label"] for record in records}
    sets = []
    for key in ("train", "validation", "test"):
        indices = fold[key]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("비어 있거나 중복된 분할입니다")
        if {records[index]["label"] for index in indices} != expected_classes:
            raise ValueError("분할에 클래스가 누락되었습니다")
        sets.append({records[index]["group"] for index in indices})
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("학습·검증·시험 사이에 그룹 누출이 있습니다")


def make_folds(records, *, n_splits=7, seed=42):
    if not records:
        raise ValueError("분할할 자료가 없습니다")
    labels = np.asarray([row["label"] for row in records])
    groups = np.asarray([row["group"] for row in records])
    classes = set(labels.tolist())
    group_classes = {}
    for group, label in zip(groups, labels):
        if group in group_classes and group_classes[group] != int(label):
            raise ValueError("현재 프로토콜은 그룹 안에서 라벨이 상충하는 자료를 허용하지 않습니다")
        group_classes[group] = int(label)
    if len(classes) < 2 or min(Counter(group_classes.values()).values()) < n_splits:
        raise ValueError("각 클래스의 독립 그룹 수가 외부 fold 수보다 적습니다")
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for number, (development, test) in enumerate(outer.split(np.zeros(len(labels)), labels, groups)):
        inner_counts = Counter(group_classes[group] for group in set(groups[development]))
        inner_count = min(5, min(inner_counts.values()))
        if inner_count < 2:
            raise ValueError("내부 검증을 나눌 그룹이 부족합니다")
        inner = StratifiedGroupKFold(n_splits=inner_count, shuffle=True, random_state=seed + number + 1)
        candidate = None
        for train_local, validation_local in inner.split(np.zeros(len(development)), labels[development], groups[development]):
            if set(labels[development[train_local]]) == classes and set(labels[development[validation_local]]) == classes:
                candidate = (development[train_local], development[validation_local])
                break
        if candidate is None:
            raise ValueError("모든 클래스를 포함하는 내부 검증 fold가 없습니다")
        fold = {"fold": number, "train": candidate[0].tolist(), "validation": candidate[1].tolist(), "test": test.tolist()}
        assert_fold(records, fold)
        folds.append(fold)
    seen = [index for fold in folds for index in fold["test"]]
    if sorted(seen) != list(range(len(records))):
        raise ValueError("외부 OOF 시험이 모든 행을 정확히 한 번 포함하지 않습니다")
    return folds
