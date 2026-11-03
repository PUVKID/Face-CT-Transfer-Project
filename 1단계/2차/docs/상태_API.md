# 실시간 상태 API 계약

평가·획득 집계 상태를 읽기 전용으로 조회한다. 프런트는 상대 URL을 사용하고 허용한 상태·로그만 반환한다.

서버는 `127.0.0.1`에만 바인딩한다. 원본 상태는 상위 `work/stage1_revision2/status.json`이며 서버가 허용한 상태·로그만 읽는다. 파일 경로나 명령을 입력하는 API는 없다. 상태 원본이 없거나 갱신 후 30초가 지나면 `stale=true`다. 다운로드·평가 중단과 상태 노후화를 구분한다.

## GET /api/status

응답은 JSON, `schema_version=1`이며 다음 키를 유지한다. 알려지지 않은 실제 측정값은 `null`이고 빈 시계열은 `[]`다. 아래는 필드의 의미이며 예시 성능을 실제값으로 생성하지 않는다.

| 키 | 형식과 의미 |
|---|---|
| `updated_at` | 실제 원자 상태 저장의 한국 시각 ISO 문자열 또는 null |
| `stale`, `status` | bool, `waiting/downloading/verifying/ready/running/complete/paused/failed` |
| `project` | `{name, revision, stage}` 문자열 |
| `current_task` | `{title, phase, detail}` 한국어 문자열 |
| `progress` | `{completed,total,unit,percent,eta_seconds,eta_is_estimate}`. 미확보/미측정은 null. ETA는 추정 여부 필수 |
| `model` | `{name,training_data,source_url,version,checkpoint_verified,license_status,reported_metrics,measured_metrics}`. metric 배열 원소 `{benchmark,metric,value,unit,provenance}` |
| `downloads` | 파일별 `{id,name,status,received_bytes,total_bytes,verified,expected_digest,digest_algorithm,bytes_per_second,eta_seconds,error}`. 상태 `pending/downloading/verifying/complete/failed` |
| `steps` | `{id,title,status,detail}` 배열. 상태 `waiting/active/complete/paused/failed` |
| `techniques` | `{id,name,status,detail}` 배열. 상태 `planned/active/unused/complete/paused/failed` |
| `evaluation` | `{status,benchmark,precision,images_completed,images_total,pairs_completed,pairs_total,fold_completed,fold_total,epoch,metrics}`. 얼굴 재현 전 metrics는 null |
| `resources` | `{gpu,disk_free_bytes}`. gpu는 null 또는 `{name,utilization_percent,memory_used_bytes,memory_total_bytes,measured_at}` |
| `processes` | `{role,pid,alive,started_at}` 배열. `alive`는 실제 OS 조회 또는 미확인 null |
| `series` | `{throughput,gpu_utilization,metrics}`. 실제 관측만 `{time,value}` 배열로 누적하며 합성값을 만들지 않음 |
| `legacy` | `{model,status,detail}`. 기존 AdaFace는 `user_stopped`이며 재개 API 없음 |
| `external_data` | `{name,status,received_bytes,verified_bytes,total_bytes,updated_at}` 배열. 기존 LUNA 수집을 읽기 전용으로 별도 표시 |
| `log_sources` | `{id,label}` 배열. `acquisition/evaluation/server`만 허용 |
| `notices` | 실제 한계·대기 이유의 한국어 문자열 배열 |

TopoFR 원보고 LFW99.87%, IJB-C TAR97.84%@FAR1e-4/96.71%@1e-5는 `reported_metrics`에 넣고 출처를 표시한다. 직접 얼굴 평가 전 `measured_metrics=[]`, `evaluation.metrics=null`이다. 원보고값을 진행 그래프나 직접 측정에 복사하지 않는다. 다운로드 완료는 길이·필요한 해시 검증 뒤만 `complete/verified=true`다. TopoFR은 공식 전체 해시 미제공을 명시하고 실제 전체 바이트의 SHA-256을 최초 획득 영수증에 고정한다.

LFW 실행 중 `evaluation.status`는 `running`, 준비 완료는 `ready`, 정렬 실패 집계 보존은 `protocol_unmet`, 실제 고정 pair 평가 완료는 `complete`다. `detection_coverage`는 성공 영상 수/전체 영상 수인 0~1 분율 또는 미측정 null이며 화면에서만 100을 곱해 %로 표시한다. `failed_images`는 비음수 정수인 실패 영상 수 또는 미측정 null이다. `protocol_unmet`의 현재 작업 설명은 실제 성공·전체·실패 수, 정확도 미측정, 조건 미충족 검토 대기를 표시하며 IJB 할당량 문구로 대체하지 않는다. 원본 이름·사진·개별 점수는 반환하지 않는다. LFW 완료 뒤 전체 상태는 IJB-C 외부 대기 때문에 `paused`일 수 있으며 LFW 직접값과 IJB-C 미측정을 동시에 구분한다. 처리량의 단위는 현재 progress의 `bytes` 또는 `images`를 따른다. 얼굴 평가에는 epoch가 null이며 학습 곡선을 만들지 않는다.

원본 상태 JSON이 배열·문자열·null 등 객체가 아니거나 알려진 중첩 객체/배열 형식이 잘못되면 진행을 정상으로 표시하지 않는다. API는 예외로 연결을 끊지 않고 `status=failed, stale=true`, 명확한 상태 검증 실패 설명, 미측정 집계로 응답한다. 비유한 수와 UTF-8 직렬화가 불가능한 문자열도 유효 상태로 제공하지 않는다.

## 알려진 필드의 명시 타입

검증 스키마는 `benchmark/status.py`의 `STATUS_SCHEMA`에 선언하며 초기값 `null`이나 빈 배열에서 타입을 추론하지 않는다. 문자열에는 객체·배열·bool을 허용하지 않는다. 숫자는 bool과 구분한 유한 int/float이고, 개수·바이트는 bool이 아닌 비음수 정수다. 미측정 nullable 값은 null 그대로 두며 false와 0을 미측정으로 바꾸지 않는다. 아래에서 `?`는 null 허용이다.

| 객체/배열 | 알려진 필드 타입 |
|---|---|
| 루트 | schema_version=int(현재1), updated_at=string?, stale=bool, status=string(정의된 상태값) |
| project/current_task/legacy | 선언한 이름·차수·단계·제목·상태·설명 필드는 string |
| progress | completed/total=개수?, unit=string, percent/eta_seconds=유한 숫자?, eta_is_estimate=bool |
| model | name/training_data/source_url/version/license_status=string, checkpoint_verified=bool, 선택 verification_detail=string |
| reported_metrics/measured_metrics 행 | benchmark/metric/unit/provenance=string, value=유한 숫자. 미측정은 빈 배열이며 잘못된 행을 미측정 행으로 정규화하지 않음 |
| downloads 행 | id/name/status=string, received_bytes/total_bytes=바이트?, verified=bool, expected_digest/digest_algorithm/error=string?, bytes_per_second/eta_seconds=유한 숫자? |
| downloads 추가 검증 필드 | official_digest_verified=bool, verification_basis=string. 이전 형식에서 생략할 수 있으며 제공된 값은 검증함 |
| steps/techniques 행 | 선언한 id·title 또는 name·status·detail은 string |
| evaluation | status/precision=string, benchmark=string?, 영상·쌍·fold·epoch·failed_images=개수?, detection_coverage=0~1 유한 숫자? |
| evaluation.metrics | null 또는 객체. 알려진 accuracy/std는 유한 숫자, author_input_equivalence는 bool |
| resources | disk_free_bytes=바이트?, gpu=null 또는 객체 |
| gpu | name/measured_at=string, utilization_percent=유한 숫자, memory_used_bytes/memory_total_bytes=바이트 |
| processes 행 | role=string, pid=bool이 아닌 양의 Windows DWORD 정수, alive=bool?, started_at=string? |
| series 각 배열 행 | time=string, value=유한 숫자. 배열이 비어 있어도 행 스키마는 유지 |
| external_data 행 | name=string, status/updated_at=string?, received_bytes/verified_bytes/total_bytes=바이트? |
| log_sources 행/notices | id/label=string인 행 배열 / string 배열 |

배열의 각 행은 위 필수 필드를 포함한다. downloads의 두 추가 검증 필드만 이전 형식과의 호환을 위해 선택 필드다. 알려진 nullable scalar에 객체나 배열을 넣거나 행의 문자열 필드에 객체를 넣으면 전체 상태를 failed/stale로 반환하고 직접 측정 집계는 null/빈 배열로 되돌린다. 정상 한국어 문자열·null·false·0 및 기존 검증 근거는 그대로 유지한다. updated_at=null은 아직 관측 시각이 없는 정상 대기 상태로서 stale=true이며 임의의 현재 시각을 만들지 않는다.

처리량 `series.throughput`은 현재 `progress.unit`의 초당 처리량이다. 새 prepare/evaluate 작업 시작과 bytes→images 등 단위 전환 때 현재 처리량 배열을 새로 시작하며 같은 단위의 연속 관측은 유지한다. 이전 관측은 비공개 `throughput_history.jsonl`에 원래 단위·전환 이유와 함께 보존한다. 지표 시계열·모델 직접 측정 기록·기존 실행 로그를 이 초기화에 포함하지 않는다. API 행 형식 `{time,value}`를 확대하지 않는다.

새 prepare/evaluate 작업 시작 직후에는 이전 진행률을 상속하지 않는다. 관측 전 progress의 completed/total/percent/eta_seconds와 이번 작업의 영상·쌍·fold 처리수/전체수는 null, unit은 빈 문자열인 단위 미확인이다. 이전 progress/evaluation 원문은 비공개 `progress_history.jsonl`에 보존하며 모델 실측 기록·지표 시계열은 유지한다. prepare의 검출 집계는 새 관측 전 null이고 evaluate의 기존 정렬 coverage/실패 수는 입력 정렬 근거로 유지하되 이번 추론 처리수와 구분한다. 실제 추론에 진입하면 검증한 고정 pair/fold 전체수를 표시한다.

기법 상태는 실제 단계 전환에 연결한다. 원본 추출 중 정렬은 `planned`이고 실제 정렬 callback부터 `active`, 전체 성공 뒤 `complete`다. 정렬 조건 미충족은 `paused`와 실제 성공·전체·실패 수·coverage를 표시하며 FP32/반전/LFW는 미적용 `planned`다. 평가 입력 검증 뒤 실제 추론 callback에서 FP32와 반전이 `active`가 되고 전체 특징 처리 뒤 LFW 계산이 `active`가 되며, 정상 평가 반환 뒤에만 이 기법들을 `complete`로 표시한다. 검증된 전체 캐시 재사용은 완료 근거와 신규 추론 없음을 구분하고 이전 정렬 완료 기록은 새 평가의 입력 검증 대기와 구분한다. 실패 종료 때 적용 중 기법은 `failed`가 되며 도달 전 계획은 미적용으로 남는다. IJB-C template 집계와 CT 학습은 이번 LFW 작업에서 `unused`다. 이 상태는 적용 이력이며 정확도 측정을 대신하지 않는다.

다운로드 원소에 `official_digest_verified`와 `verification_basis`를 추가한다. `verified=true`만으로 공식 체크섬 대조나 모델 적재 검증을 뜻하지 않는다. `official_sha256`은 공식 SHA-256 대조, `official_git_blob_sha1`은 고정 저자 소스 객체 대조, `official_size_and_first_acquisition_sha256_no_published_digest`는 공식 길이·최초 로컬 SHA 고정이다. 화면은 이 근거를 구분해야 한다. `model.checkpoint_verified`는 안전 역직렬화·R200 strict 적재·tensor 유한값 검사 전 false다.

공식 MS1MV2 입력 작업은 현재 작업 제목과 alignment 기법명·단계명에 **제공된 정렬 입력 검증**을 명시한다. header·decode 검증 중 active, 전체 성공 또는 결합 검증된 캐시 재사용 뒤 complete이며 이번 작업이 얼굴 검출·5점 정렬을 수행했다고 표시하지 않는다. 검출률과 검출실패수는 해당 없음인 null이고 처리 영상 수만 누적한다. 이전 YuNet coverage를 새 입력 종류에 상속하지 않는다. FP32/flip/LFW 상태는 기존 실제 추론·계산·완료 전환을 따른다.

고정 원본·완료 영수증·SHA/CRC 검증 callback 뒤에만 downloads에공식 bin64738112바이트 완료 행을 추가한다. `official_digest_verified=false`이며 최초 로컬 SHA와 ZIP CRC 대조라는 verification_basis를 유지한다. 네트워크 수신중이나 얼굴 정확도 검증 완료로 표시하지 않는다. input-kind별 LFW 측정 기록은 서로 다른 metric 설명으로 보존한다.

## GET /api/logs?id=acquisition&limit=100

허용 id는 `acquisition/evaluation/server`다. limit은 정수 1~200이며 **JSON 직렬화와 이스케이프가 끝난 HTTP 본문 전체의 UTF-8 바이트 수**가 최대 65536이다. 응답은 `{source,lines,truncated,updated_at}`이고 lines는 문자열 배열이다. 원문의 빈 줄을 유지하고 크기·줄 수 때문에 생략하면 `truncated=true`다. 상한을 넘으면 오래된 완전한 줄부터 제외하며 한 줄만으로도 상한을 넘으면 그 줄을 전부 제외하므로 lines가 빈 배열일 수 있다. 문자열·한글 바이트·JSON 본문을 중간에서 자르지 않는다. 비공개 원본 실패·인증값·쿼리 토큰·개별 얼굴 특징을 이 API 로그로 노출하지 않는다. 알려지지 않은 id·중복/기타 query 키·잘못된 limit은 400이다. POST/PUT/DELETE와 다른 API는 지원하지 않는다.

## 정적 파일

`dashboard/frontend/dist` 안의 빌드 파일만 제공한다. 심볼릭 링크·경로 이탈·숨김 파일·정적 root 외부 파일은 거부한다. `/api` 오류에 SPA HTML을 반환하지 않는다. 접근 로그는 고정 운영 경로에 남기고 URL query 원문을 출력하지 않는다.
