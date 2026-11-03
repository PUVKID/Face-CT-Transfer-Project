"""원보고 재현 규칙과 보수적인 실제 FAR를 함께 계산합니다."""
import numpy as np


def verification_roc(scores, labels, fars=(1e-4, 1e-5, 1e-6)):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores) or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("검증 점수·이진 라벨이 유효하지 않습니다")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("동일인·타인 쌍이 모두 필요합니다")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels[order], dtype=np.int64)[ends]]
    fp = np.r_[0, 1 + ends - tp[1:]]
    tpr, fpr = tp / positives, fp / negatives
    thresholds = np.r_[np.inf, ordered[ends]]
    result = []
    for far in fars:
        if not np.isfinite(far) or not 0 <= far <= 1:
            raise ValueError("FAR 범위가 유효하지 않습니다")
        eligible = np.flatnonzero(fpr <= far)
        best = eligible[np.argmax(tpr[eligible])]
        # 원저자 코드는 ROC를 뒤집어 동률 FAR에서 큰 TAR를 고릅니다.
        from sklearn.metrics import roc_curve
        original_fpr, original_tpr, original_thresholds = roc_curve(labels, scores)
        original_fpr, original_tpr, original_thresholds = original_fpr[::-1], original_tpr[::-1], original_thresholds[::-1]
        original = int(np.argmin(np.abs(original_fpr - far)))
        result.append({"target_far": float(far), "tar": float(tpr[best]), "actual_far": float(fpr[best]), "threshold": None if np.isinf(thresholds[best]) else float(thresholds[best]), "original_nearest_tar": float(original_tpr[original]), "original_nearest_actual_far": float(original_fpr[original]), "original_nearest_threshold": None if np.isinf(original_thresholds[original]) else float(original_thresholds[original]), "genuine_pairs": positives, "impostor_pairs": negatives})
    return result


def lfw_ten_fold(scores, labels, input_kind="cosine"):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != (6000,) or labels.shape != (6000,) or not np.isfinite(scores).all() or input_kind not in {"cosine", "squared_l2"}:
        raise ValueError("LFW 고정 6000쌍의 유한 코사인 점수가 필요합니다")
    if input_kind == "cosine":
        if np.any(np.abs(scores) > 1 + 1e-6):
            raise ValueError("코사인 범위가 유효하지 않습니다")
        distances = 2 - 2 * np.clip(scores, -1, 1)
    else:
        if np.any((scores < 0) | (scores > 4 + 1e-6)):
            raise ValueError("정규화 특징의 squared-L2 범위가 유효하지 않습니다")
        distances = scores
    thresholds = np.arange(0, 4, 0.01)
    folds = []
    indices = np.arange(6000)
    for fold in range(10):
        test = (indices >= fold * 600) & (indices < (fold + 1) * 600)
        train = ~test
        correct = ((distances[train, None] < thresholds[None, :]) == labels[train, None]).mean(axis=0)
        threshold = thresholds[int(np.argmax(correct))]
        accuracy = float(((distances[test] < threshold) == labels[test]).mean())
        folds.append({"fold": fold, "accuracy": accuracy, "threshold": float(threshold), "training_pairs": 5400, "test_pairs": 600})
    accuracies = [fold["accuracy"] for fold in folds]
    return {"accuracy": float(np.mean(accuracies)), "std": float(np.std(accuracies)), "pairs": 6000, "folds": folds, "metric": "고정 pair 10fold 검증 정확도", "threshold_policy": "나머지 9fold의 squared-L2 정확도 최대; 첫 동률 임계값"}


def aggregate_templates(features, templates, medias, faceness, flipped=None):
    values = np.asarray(features, dtype=np.float64)
    templates, medias = np.asarray(templates), np.asarray(medias)
    faceness = np.asarray(faceness, dtype=np.float64)
    if values.ndim != 2 or templates.shape != (len(values),) or medias.shape != templates.shape or faceness.shape != templates.shape:
        raise ValueError("template·media·특징 수가 다릅니다")
    if flipped is not None:
        flipped = np.asarray(flipped, dtype=np.float64)
        if flipped.shape != values.shape:
            raise ValueError("좌우반전 특징 형상이 다릅니다")
        values = values + flipped
    if not np.isfinite(values).all() or not np.isfinite(faceness).all() or np.any((faceness < 0) | (faceness > 1)):
        raise ValueError("특징·얼굴 검출 점수가 유효하지 않습니다")
    values = values * faceness[:, None]
    unique = np.unique(templates)
    output = []
    for template in unique:
        selected = templates == template
        local = values[selected]
        media = medias[selected]
        combined = np.sum([local[media == item].mean(axis=0) for item in np.unique(media)], axis=0)
        norm = np.linalg.norm(combined)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("0 또는 비유한 template 특징입니다")
        output.append(combined / norm)
    return unique, np.asarray(output, dtype=np.float32)
