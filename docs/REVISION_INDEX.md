# 🗂️ 차수와 검토 회차

실험 단계·파일 차수·검토 회차를 구분합니다. 1단계 1차의 합성 진단과 2단계 1차의 학습 전 준비 검증을 구분합니다. 실제 분류 학습은 06:46 시작 후 06:48 오류로 중단됐으며 수리 후 36검사·새 배치 16 측정은 통과했으나 독립 재검증 36개 검사 통과 후 07:12:34에 재시작했습니다. 07:12 시점에는 특징 계산 중이며 IQ 완료 결과는 0/30이고 3단계·최종본도 미완료입니다.

| 차수 | 자료 폴더 | 상세 설명 | 직전 비교 |
|---|---|---|---|
| 1단계 1차 | [폴더](../1단계/1차) | [총괄 보고서](../1단계/1차/reports/총괄_보고서.md) | [최초 전체 비교](INITIAL_DIFF.md) |
| 2단계 1차 | [폴더](../2단계/1차) | [학습 전 준비 검증·재시작 확인](../2단계/1차/README.md) | [직전 공개본 이후 전체 비교](STAGE2_PREPARATION_DIFF.md) |

1단계는 최초 산출물로 초기 README와 비교하며, 2단계는 직전 1단계 공개 커밋 이후의 실제 전체 변경과 비교합니다. 아래 30개 회차는 1단계 합성 검토이며 2단계 학습 완료 수가 아닙니다.

| 검토 회차 | 보고서 | 직접 결과 |
|---|---|---|
| 01 | [보고서](../1단계/1차/reports/01_rgb_endpoints.md) | [JSON](../1단계/1차/results/01_rgb_endpoints.json) |
| 02 | [보고서](../1단계/1차/reports/02_grayscale_replication.md) | [JSON](../1단계/1차/results/02_grayscale_replication.json) |
| 03 | [보고서](../1단계/1차/reports/03_resize_non_square.md) | [JSON](../1단계/1차/results/03_resize_non_square.json) |
| 04 | [보고서](../1단계/1차/reports/04_bgr_explicit.md) | [JSON](../1단계/1차/results/04_bgr_explicit.json) |
| 05 | [보고서](../1단계/1차/reports/05_float_normalization.md) | [JSON](../1단계/1차/results/05_float_normalization.json) |
| 06 | [보고서](../1단계/1차/reports/06_reject_nan.md) | [JSON](../1단계/1차/results/06_reject_nan.json) |
| 07 | [보고서](../1단계/1차/reports/07_reject_infinity.md) | [JSON](../1단계/1차/results/07_reject_infinity.json) |
| 08 | [보고서](../1단계/1차/reports/08_reject_range.md) | [JSON](../1단계/1차/results/08_reject_range.json) |
| 09 | [보고서](../1단계/1차/reports/09_reject_channels.md) | [JSON](../1단계/1차/results/09_reject_channels.json) |
| 10 | [보고서](../1단계/1차/reports/10_reject_shape.md) | [JSON](../1단계/1차/results/10_reject_shape.json) |
| 11 | [보고서](../1단계/1차/reports/11_inference_single.md) | [JSON](../1단계/1차/results/11_inference_single.json) |
| 12 | [보고서](../1단계/1차/reports/12_inference_batch.md) | [JSON](../1단계/1차/results/12_inference_batch.json) |
| 13 | [보고서](../1단계/1차/reports/13_batch_independence.md) | [JSON](../1단계/1차/results/13_batch_independence.json) |
| 14 | [보고서](../1단계/1차/reports/14_deterministic_repeat.md) | [JSON](../1단계/1차/results/14_deterministic_repeat.json) |
| 15 | [보고서](../1단계/1차/reports/15_input_sensitivity.md) | [JSON](../1단계/1차/results/15_input_sensitivity.json) |
| 16 | [보고서](../1단계/1차/reports/16_frozen_gradient.md) | [JSON](../1단계/1차/results/16_frozen_gradient.json) |
| 17 | [보고서](../1단계/1차/reports/17_frozen_bn.md) | [JSON](../1단계/1차/results/17_frozen_bn.json) |
| 18 | [보고서](../1단계/1차/reports/18_head_gradient.md) | [JSON](../1단계/1차/results/18_head_gradient.json) |
| 19 | [보고서](../1단계/1차/reports/19_head_update.md) | [JSON](../1단계/1차/results/19_head_update.json) |
| 20 | [보고서](../1단계/1차/reports/20_unfrozen_gradient.md) | [JSON](../1단계/1차/results/20_unfrozen_gradient.json) |
| 21 | [보고서](../1단계/1차/reports/21_refreeze.md) | [JSON](../1단계/1차/results/21_refreeze.json) |
| 22 | [보고서](../1단계/1차/reports/22_checkpoint_roundtrip.md) | [JSON](../1단계/1차/results/22_checkpoint_roundtrip.json) |
| 23 | [보고서](../1단계/1차/reports/23_missing_checkpoint.md) | [JSON](../1단계/1차/results/23_missing_checkpoint.json) |
| 24 | [보고서](../1단계/1차/reports/24_mismatch_checkpoint.md) | [JSON](../1단계/1차/results/24_mismatch_checkpoint.json) |
| 25 | [보고서](../1단계/1차/reports/25_seed_reproduction.md) | [JSON](../1단계/1차/results/25_seed_reproduction.json) |
| 26 | [보고서](../1단계/1차/reports/26_cuda_fp32.md) | [JSON](../1단계/1차/results/26_cuda_fp32.json) |
| 27 | [보고서](../1단계/1차/reports/27_cuda_fp16.md) | [JSON](../1단계/1차/results/27_cuda_fp16.json) |
| 28 | [보고서](../1단계/1차/reports/28_cuda_bfloat16.md) | [JSON](../1단계/1차/results/28_cuda_bfloat16.json) |
| 29 | [보고서](../1단계/1차/reports/29_channels_last.md) | [JSON](../1단계/1차/results/29_channels_last.json) |
| 30 | [보고서](../1단계/1차/reports/30_cpu_cuda_parity.md) | [JSON](../1단계/1차/results/30_cpu_cuda_parity.json) |
