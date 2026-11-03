# 🔎 1단계 2차 · 실제 전체 변경

> **현재 관측:** YuNet 독립 정렬은 7,701개 중 7,631개 성공·70개 실패로 평가조건 미충족입니다. 공식 정렬 LFW 파일의 획득·정적 검사는 완료했지만 영상 해석·TopoFR GPU 추론·얼굴 정확도는 아직 미측정입니다.


직전 공개 기준은 `09d166f34a7d1ce9e65973f596908396231d8569`입니다. [공백·주석을 포함한 전체 Git 비교](https://github.com/PUVKID/Face-CT-Transfer-Project/compare/09d166f34a7d1ce9e65973f596908396231d8569...main)를 연결합니다. 이후 main 변경으로 비교가 넓어질 수 있으므로 아래 동결 파일 목록과 함께 확인합니다. 같은 단계의 직전 차수는 1단계 1차이며 이전 공개 기준에는 2단계 1차도 포함됩니다.

## 실제 구현과 한계

새 TopoFR 공개 버전의 획득·원보고·평가 프로토콜·안전 경계, 읽기 전용 API, 실제 WebGL 대시보드·상태·로그·서체를 추가합니다. 프런트 형식·프로세스 문구·로그 회귀 검사 19개와 빌드·실제 CUA 조작은 통과했지만 얼굴 정확도는 미측정입니다. 원보고 IJB-C FAR 조건과 보조 LFW를 구분합니다. 가중치의 최초 로컬 SHA 고정은 공식 체크섬 대조가 아니며 첫 strict CPU 적재의 필수 인자 오류는 보존했습니다. 저자 설정 복원 뒤 12:14:53 CPU strict 적재·1,808개 텐서 검사·백엔드 검사 19개가 통과했지만 GPU 얼굴 정확도는 미측정입니다. IJB 배포는 외부 할당량 초과 HTML로 차단됐습니다. 모바일·모션 축소 환경의 실제 렌더는 미확인입니다.

기존 1단계 1차·2단계 1차 하위 파일은 원문을 유지했습니다. 이전 AdaFace CT의 사용자 중단과 부분 결과 보존을 최신 통합 설명에 반영합니다. 원본 보고서를 완성 논문이나 새 얼굴 모델의 완료 결과로 바꾸지 않습니다. 무의미한 30개 보고서·학습·결과 커밋은 만들지 않습니다.

## 전체 파일 변화

추가·수정 경로는 아래와 같고 삭제는 없습니다. 소스 내용 전체·주석·공백·개행은 기준 원문과 대조한 비공개 전체 diff에 보존하며 원격 전체 비교에서도 확인할 수 있습니다. API·디자인 문서의 작업자·개인 승인 참조 표현은 공개용 기술 설명으로 정제했고 프런트 설치 캐시 경로는 같은 프로젝트 내부 상대 경로로 옮겼습니다. 원본은 수정하지 않았고 변환 전후 해시·전체 diff를 별도로 보존합니다.

| 파일 | 변경 |
|---|---|
| `1단계/2차/README.md` | 추가 |
| `1단계/2차/benchmark/__init__.py` | 추가 |
| `1단계/2차/benchmark/acquisition.py` | 추가 |
| `1단계/2차/benchmark/evaluation.py` | 추가 |
| `1단계/2차/benchmark/metrics.py` | 추가 |
| `1단계/2차/benchmark/model.py` | 추가 |
| `1단계/2차/benchmark/protocol.py` | 추가 |
| `1단계/2차/benchmark/safety.py` | 추가 |
| `1단계/2차/benchmark/sources.json` | 추가 |
| `1단계/2차/benchmark/status.py` | 추가 |
| `1단계/2차/dashboard/frontend/index.html` | 추가 |
| `1단계/2차/dashboard/frontend/package.json` | 추가 |
| `1단계/2차/dashboard/frontend/pnpm-lock.yaml` | 추가 |
| `1단계/2차/dashboard/frontend/pnpm-workspace.yaml` | 추가 |
| `1단계/2차/dashboard/frontend/public/fonts/NanumGothic-Bold.ttf` | 추가 |
| `1단계/2차/dashboard/frontend/public/fonts/OFL.txt` | 추가 |
| `1단계/2차/dashboard/frontend/src/App.jsx` | 추가 |
| `1단계/2차/dashboard/frontend/src/Icon.jsx` | 추가 |
| `1단계/2차/dashboard/frontend/src/Panels.jsx` | 추가 |
| `1단계/2차/dashboard/frontend/src/Workflow.jsx` | 추가 |
| `1단계/2차/dashboard/frontend/src/format.js` | 추가 |
| `1단계/2차/dashboard/frontend/src/format.test.js` | 추가 |
| `1단계/2차/dashboard/frontend/src/main.jsx` | 추가 |
| `1단계/2차/dashboard/frontend/src/styles.css` | 추가 |
| `1단계/2차/dashboard/frontend/src/useLogs.test.js` | 추가 |
| `1단계/2차/dashboard/frontend/src/useStatus.js` | 추가 |
| `1단계/2차/dashboard/frontend/vite.config.js` | 추가 |
| `1단계/2차/dashboard/server.py` | 추가 |
| `1단계/2차/docs/공식_LFW_획득.md` | 추가 |
| `1단계/2차/docs/대시보드_디자인.md` | 추가 |
| `1단계/2차/docs/상태_API.md` | 추가 |
| `1단계/2차/docs/얼굴_평가_프로토콜.md` | 추가 |
| `1단계/2차/docs/요구사항.md` | 추가 |
| `1단계/2차/docs/획득과_실행.md` | 추가 |
| `1단계/2차/reports/lfw_official_acquisition.json` | 추가 |
| `1단계/2차/reports/lfw_yunet_coverage.json` | 추가 |
| `1단계/2차/reports/모델_선정과_현재_진행.md` | 추가 |
| `1단계/2차/tests/test_acquisition_safety.py` | 추가 |
| `1단계/2차/tests/test_evaluation_runner.py` | 추가 |
| `1단계/2차/tests/test_face_metrics.py` | 추가 |
| `1단계/2차/tests/test_model_protocol.py` | 추가 |
| `1단계/2차/tests/test_official_lfw_acquisition.py` | 추가 |
| `1단계/2차/tests/test_status_schema.py` | 추가 |
| `1단계/2차/tests/test_status_server.py` | 추가 |
| `1단계/2차/tools/acquire_official_lfw.py` | 추가 |
| `1단계/2차/tools/inspect_model.py` | 추가 |
| `1단계/2차/tools/official_lfw_range_member.py` | 추가 |
| `1단계/2차/tools/run_acquisition.py` | 추가 |
| `1단계/2차/tools/run_lfw.py` | 추가 |
| `CHANGELOG.md` | 수정 |
| `README.md` | 수정 |
| `docs/ALIGNED_LFW_SOURCE_STATUS.md` | 추가 |
| `docs/REVISION_INDEX.md` | 수정 |
| `docs/STAGE1_REVISION2_DIFF.md` | 추가 |

## 바이너리와 제외 범위

글꼴 바이너리 `1단계/2차/dashboard/frontend/public/fonts/NanumGothic-Bold.ttf`: 2,073,868바이트, SHA-256 `f96298f9fb18e364d2370f4c3ce948ac67a2b61af992d7234bc15c42b033c674`. OFL을 함께 보존했습니다. 바이너리 내용을 텍스트 차이로 추정하지 않습니다. 다른 모델·원영상·개별 예측·임베딩·가중치·vendor 다운로드·의존성·캐시·운영 로그·개인 지침은 공개하지 않습니다. 실제 제외 디렉터리와 선택 범위는 별도 비공개 기록에 남겼습니다.

이 차수는 기술 진행 기록이며 논문 원고가 없어 동일 원고 PDF는 적용하지 않습니다. 코드 실행 전 독립 검증 뒤 실제 얼굴 평가 결과를 기다리는 공개 준비 초안이므로 최종 독립 검토·커밋·원격 일치 전 완료로 집계하지 않습니다.


## 🔎 검증된 준비와 실제 관측 — 2026-11-03

코드 실행 전 독립 검증에서 Python72개·Node19개 검사와 프런트 빌드가 통과했습니다. 이 공개 코드 판본은 당시 검증된 백엔드26파일 보존본과 프런트18파일을 사용합니다. 이후 개발 중인 제한 primitive reader와 새 검사는 포함하지 않습니다. 코드 검사는 얼굴 성능 재현 결과가 아닙니다.

CPU 가중치 검사는 weights_only=True·strict=True로1,808개 텐서와303,272,704개 파라미터를 적재했습니다. 원보고 IJB-C TAR97.84%@FAR=1e-4·96.71%@FAR=1e-5와 LFW99.87%는 저자 보고이며 직접 측정값이 아닙니다. 공개 후보의 같은 조건에서 선택한 모델로, 모든 상용·미공개 모델을 포함한 절대 최고를 주장하지 않습니다.

첫 실제 준비는 원본 추출 뒤 한글 ONNX 파일명 로딩에서 실패했습니다. 같은 공식 SHA의 바이트를 공식 buffer API에 전달하도록 보완했습니다. 두 번째는 정렬3975/7701 관측 중 상태 파일 원자 교체의 WinError5로 실패했습니다. 제한된 일시 점유 재시도로 보완했으며 실제 점유 주체는 미확정입니다. 과거75초 정식 검사 시간 초과는 실패로 보존했고 후속 정식 검사는 통과했지만 당시 원인은 확정하지 않았습니다.

세 번째 실제 준비는14:43:39~14:45:15 전체7,701개를 검사해7,631개 성공·70개 실패·다중 얼굴511개를 기록했습니다. 검출·정렬 성공률99.0910%는 얼굴 인식 정확도가 아닙니다. 고정6,000쌍을 모두 준비하지 못해 protocol_unmet으로 종료했고 실패쌍을 삭제하거나 임의 resize로 대체하지 않았습니다. TopoFR GPU 추론과 얼굴 정확도 산출은0회입니다.

공식 InsightFace MS1MV2 ZIP의 faces_emore/lfw.bin 한 파일을 정상206 범위 응답으로 획득했습니다. 해제 크기64,738,112바이트·CRC32 2d7d0c73·DEFLATE 종료·남는 데이터 검사를 확인했습니다. 최초 로컬 SHA-256은9b711dca71a47fc9f295a28514abe8cc4e457632a8d634264691398c99569593이며 저자의 별도 게시 SHA 대조가 아닙니다. 실행 없는 정적 검사에서30,044개 opcode·영상 문자열12,000개·참/거짓 각3,000개·GLOBAL/REDUCE0·후행0을 확인했습니다. 일반 pickle 역직렬화·영상 decode·인식 평가를 완료했다는 뜻은 아닙니다. TopoFR 저자 평가 당시 바이트와 동일성은 미확인입니다.

IJB-C는 외부 서비스 할당량 대기입니다. 기존 AdaFace CT 실행은 사용자 중단·보존 상태이며 새 모델 결과로 섞지 않습니다. 모바일·모션 축소 런타임 검증은 미확인입니다. 데스크톱에서는 실제 WebGL 정지/재생·단계 선택·로그·실측 빈 상태와 영상/초 레이블 및 처리량159관측을 접근성 트리로 확인했습니다. 해당 상단 스크린샷은 하단 레이블의 시각 증거와 구분합니다.

이 자료는 코드·설정·집계 기술 진행 기록입니다. 개별 검사나 수리를30개 연구·30회 학습으로 세지 않습니다. 논문 원고가 없으므로 논문 PDF는 적용하지 않습니다. 실제 얼굴 성능 평가와 후속 CT 연구는 미완료입니다.
