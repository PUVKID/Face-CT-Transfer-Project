# 🗂️ 차수와 검토 회차

> **최신 실측:** LFW6,000쌍·10fold 정확도99.85%, 저자 보고99.87% 대비−0.02%p. 독립 재계산 일치. IJB-C는 미측정이며 입력 동일성의 한계가 남아 있습니다.

[실제 평가 상세](../1단계/2차/reports/TopoFR_LFW_실제_평가_결과.md) · [기계 판독 집계](../1단계/2차/reports/lfw_official_result.json) · [모델 선정과 시점별 기록](../1단계/2차/reports/모델_선정과_현재_진행.md)

## 📊 실제 LFW 평가 결과 — 2026-11-03

TopoFR-R200 Glint360K를 공식 MS1MV2 제공 정렬 LFW의6,000쌍으로 평가해10fold 평균 정확도 **99.85%**를 얻었습니다. 정답5,991·오답9(동일인 오거부9·타인 오수락0), fold 표준편차0.203443%p(ddof=0)입니다. 표준편차는 신뢰구간이 아닙니다. 저자 보고99.87%와 차이는−0.02%p입니다. 영상 입력12,000회는 중복을 포함하며 고유 영상·인물12,000개를 뜻하지 않습니다.

실제 GPU 평가는15:52:20~16:01:53 한국 시각에 FP32·batch4·원본/좌우반전 특징 합으로 완료했습니다. 시험 fold와 분리한5,400쌍에서 임계값을 고르고6,000쌍을 한 번씩 평가했습니다. 재학습·삭제쌍·결과에 맞춘 전처리 변경 없이 저장 특징에서 구현자와 다른 검증자가 별도 NumPy 거리·임계값·10fold 수치를 재계산해 일치를 확인했습니다.

저자 당시 bin의 전체 SHA와 MXNet/OpenCV 디코딩의 픽셀 동일성은 미확인입니다. 따라서 완전히 같은 입력 재현·통계적 동등성·세계 절대 최고를 주장하지 않습니다. 주 비교 지표인 **IJB-C TAR97.84%@FAR=1e-4는 계속 저자 보고**이며 IJB-C는 외부 할당량 대기입니다. LFW 실측으로 IJB-C·의료/임상·전체 단계 완료를 대체하지 않습니다.

같은1단계2차의 후속 수정으로 제한 primitive 로더·공식 입력 캐시·평가 실행과 관련 검사·문서를 반영했습니다. 새 차수나30회 연구를 만들지 않았습니다. 원형26파일과 기존 YuNet70실패 집계는 보존하고 freeze10 백엔드28파일·동일프런트18파일을 사용합니다. 실제 얼굴/CT·개별 특징·예측·점수·가중치·운영 로그는 공개하지 않습니다. 논문 원고 없이 기술 진행 보고서와 익명 집계만 공개합니다.

## 🗃️ 직전 준비 공개 시점의 기록

아래 미측정·미실행·개발 보류 문구는 준비 커밋f221d44 시점의 기록입니다. 위 실측 및 후속 검증 결과와 구분합니다.



1단계 2차는 새 TopoFR 얼굴 벤치마크 재현과 실시간 대시보드입니다. 직접 얼굴 성능은 미측정이며 IJB 외부 할당량 대기·CPU strict 적재 검사 통과와 GPU 얼굴 평가 미측정을 구분합니다. 이전 AdaFace CT 실행은 사용자 중단 상태이고 과거 문서는 당시 관측입니다. 3단계·최종본은 미완료입니다.

| 차수 | 자료 폴더 | 상세 설명 | 직전 비교 |
|---|---|---|---|
| 1단계 1차 | [폴더](../1단계/1차) | [총괄 보고서](../1단계/1차/reports/총괄_보고서.md) | [최초 전체 비교](INITIAL_DIFF.md) |
| 2단계 1차 | [폴더](../2단계/1차) | [학습 전 준비 검증·재시작 확인](../2단계/1차/README.md) | [직전 공개본 이후 전체 비교](STAGE2_PREPARATION_DIFF.md) |
| 1단계 2차 | [폴더](../1단계/2차) | [새 얼굴 모델·LFW 실측·대시보드](../1단계/2차/README.md) | [새 차수 전체 비교](STAGE1_REVISION2_DIFF.md) |

1단계는 최초 산출물로 초기 README와 비교하며, 2단계는 직전 1단계 공개 커밋 이후의 실제 전체 변경과 비교합니다. 아래 30개 회차는 보존한 1단계 1차의 합성 검토이며 새 얼굴 모델의 재현 횟수나 2단계 학습 완료 수가 아닙니다.

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


## 🔎 검증된 준비와 실제 관측 — 2026-11-03

코드 실행 전 독립 검증에서 Python72개·Node19개 검사와 프런트 빌드가 통과했습니다. 이 공개 코드 판본은 당시 검증된 백엔드26파일 보존본과 프런트18파일을 사용합니다. 이후 개발 중인 제한 primitive reader와 새 검사는 포함하지 않습니다. 코드 검사는 얼굴 성능 재현 결과가 아닙니다.

CPU 가중치 검사는 weights_only=True·strict=True로1,808개 텐서와303,272,704개 파라미터를 적재했습니다. 원보고 IJB-C TAR97.84%@FAR=1e-4·96.71%@FAR=1e-5와 LFW99.87%는 저자 보고이며 직접 측정값이 아닙니다. 공개 후보의 같은 조건에서 선택한 모델로, 모든 상용·미공개 모델을 포함한 절대 최고를 주장하지 않습니다.

첫 실제 준비는 원본 추출 뒤 한글 ONNX 파일명 로딩에서 실패했습니다. 같은 공식 SHA의 바이트를 공식 buffer API에 전달하도록 보완했습니다. 두 번째는 정렬3975/7701 관측 중 상태 파일 원자 교체의 WinError5로 실패했습니다. 제한된 일시 점유 재시도로 보완했으며 실제 점유 주체는 미확정입니다. 과거75초 정식 검사 시간 초과는 실패로 보존했고 후속 정식 검사는 통과했지만 당시 원인은 확정하지 않았습니다.

세 번째 실제 준비는14:43:39~14:45:15 전체7,701개를 검사해7,631개 성공·70개 실패·다중 얼굴511개를 기록했습니다. 검출·정렬 성공률99.0910%는 얼굴 인식 정확도가 아닙니다. 고정6,000쌍을 모두 준비하지 못해 protocol_unmet으로 종료했고 실패쌍을 삭제하거나 임의 resize로 대체하지 않았습니다. TopoFR GPU 추론과 얼굴 정확도 산출은0회입니다.

공식 InsightFace MS1MV2 ZIP의 faces_emore/lfw.bin 한 파일을 정상206 범위 응답으로 획득했습니다. 해제 크기64,738,112바이트·CRC32 2d7d0c73·DEFLATE 종료·남는 데이터 검사를 확인했습니다. 최초 로컬 SHA-256은9b711dca71a47fc9f295a28514abe8cc4e457632a8d634264691398c99569593이며 저자의 별도 게시 SHA 대조가 아닙니다. 실행 없는 정적 검사에서30,044개 opcode·영상 문자열12,000개·참/거짓 각3,000개·GLOBAL/REDUCE0·후행0을 확인했습니다. 일반 pickle 역직렬화·영상 decode·인식 평가를 완료했다는 뜻은 아닙니다. TopoFR 저자 평가 당시 바이트와 동일성은 미확인입니다.

IJB-C는 외부 서비스 할당량 대기입니다. 기존 AdaFace CT 실행은 사용자 중단·보존 상태이며 새 모델 결과로 섞지 않습니다. 모바일·모션 축소 런타임 검증은 미확인입니다. 데스크톱에서는 실제 WebGL 정지/재생·단계 선택·로그·실측 빈 상태와 영상/초 레이블 및 처리량159관측을 접근성 트리로 확인했습니다. 해당 상단 스크린샷은 하단 레이블의 시각 증거와 구분합니다.

이 자료는 코드·설정·집계 기술 진행 기록입니다. 개별 검사나 수리를30개 연구·30회 학습으로 세지 않습니다. 논문 원고가 없으므로 논문 PDF는 적용하지 않습니다. 실제 얼굴 성능 평가와 후속 CT 연구는 미완료입니다.
