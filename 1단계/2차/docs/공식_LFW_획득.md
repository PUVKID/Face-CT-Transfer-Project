# 공식 ZIP의 단일 LFW 파일 획득

TopoFR와 InsightFace의 공식 연결에서 확인한 MS1MV2 ZIP의 `faces_emore/lfw.bin` 한 파일만 획득한다. 전체 ZIP이나 다른 mirror는 받지 않는다. 공식 원출처의 정상 공개 확인 form/206 응답과 ZIP64 메타데이터 영수증3개의 고정 SHA를 검사한다. 실제 다운로드 URL·query·확인 필드는 비공개 운영 자료로 유지하며 코드·대시보드·공개 로그에 복제하지 않는다.

고정 archive는13247758075바이트, 대상 압축48100725바이트·출력64738112바이트·CRC32 `2d7d0c73`다. 단일 일반파일의 Unix mode와 DOS directory attribute를 함께 검사하며 local/central header의 이름·flags·method·CRC·ZIP64 크기·다음 항목 경계를 대조한다. 기존 결과·부분·실패·영수증 파일을 덮어쓰거나 자동 재결합하지 않는다.

`tools/acquire_official_lfw.py`는 `--source-evidence-dir`와 `--output`을 받는다. 출력은 현재 프로젝트 `1단계/data/face_benchmarks/raw/lfw_official_ms1mv2.bin`으로 제한한다. `--execute`를 생략하면 네트워크0인 계획만 출력한다. 실제 실행은 소스 동결과 별도 독립 검토 뒤 실행한다. 실행 명령의 실제 절대경로와 운영 문맥은 비공개 인계에 기록한다.

정확한206/Content-Range/Content-Length/본문 길이·HTTPS/host·무redirect·바이트/요청 상한을 적용하고403/quota/HTML/200은 우회나 재시도 없이 중단한다. 개별 socket timeout5초와 읽기 전후의180초 soft deadline을 유지한다. 같은 response의 blocking read나 파일IO를 soft deadline만으로 강제 중단한다고 주장하지 않는다.

운영 진입점은 실제 base Python의 단일 worker를 새 Windows Job Object에 연결한 뒤 START를 보낸다. Job 배정 전 worker는 stdin에서 대기하고 네트워크 작업을 시작하지 않는다. 감독 실행기가190초에 해당 Job을 종료하고 소속 프로세스의 종료를 확인한다. 임의 PID·다른 Python·GPU 작업을 종료하지 않는다. Job 종료·OS 스케줄링·영수증 기록까지 정확히190.000초에 끝난다는 보장은 아니다. timeout 뒤 부분 파일을 유지하고 비공개 실패 영수증을 원자 저장한다. worker 종료0도 실제 출력/완료 영수증/전체 SHA를 별도로 대조하기 전 획득 완료로 표시하지 않는다.

해제는 raw DEFLATE의 출력 상한·정확한 크기·CRC·EOF·남는 데이터를 검사한다. 최초 SHA는 로컬 획득 고정이며 저자 공식 게시 SHA가 아니다. 이 단계는 pickle 실행·영상 decode·얼굴/GPU 평가를 하지 않는다. 후속은 opcode 정적 검사와 제한 primitive reader이며 `pickle.load/loads`로 임의 코드를 실행하지 않는다. 공식 MS1MV2의 LFW 출처와 TopoFR 저자 평가 당시 입력 바이트의 동일성은 별도로 검증해야 한다.

검사는 합성 ZIP64·로컬 HTTP·느린 전송·실제 Windows 자기 worker 종료·부분 보존으로 수행한다. 비공개 원형과 초기 예상 RED/독립 실패를 보존하며 기존22개 core 파일과 평가 기준은 수정하지 않는다.

원출처: https://docs.python.org/3.12/library/zipfile.html , https://docs.python.org/3.12/library/zlib.html , https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects , https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-terminatejobobject

## 획득 결과와 제한 primitive 로더

정상 단일 member 획득은64738112바이트·CRC32 `2d7d0c73`, 최초 로컬 SHA-256 `9b711dca71a47fc9f295a28514abe8cc4e457632a8d634264691398c99569593`로 완료됐다. 이는 저자가 게시한 전체 SHA 대조가 아니다. 완료 영수증 SHA도 고정하며 비공개 URL·확인 필드는 공개하지 않는다.

실제 정적 검사에서 PROTO2·EMPTY_LIST·MARK·APPENDS·BINSTRING·BINPUT·LONG_BINPUT·NEWTRUE·NEWFALSE·TUPLE2·STOP만 관측했다. 총30044 opcode, trailing0, GLOBAL/REDUCE0이었다. `benchmark/lfw_primitive.py`는 이11개 외 opcode를 거부하고 함수·객체를 역직렬화하지 않는다. PROTO/STOP 위치, stack12010·MARK4·memo12003·list2, opcode40000·각 encoded 영상16384바이트·누적64738112바이트 상한, 연속·중복 없는 memo index와 폭을 검사한다. 최종 구조는 tuple2(list[bytes]12000, list[bool]6000)만 허용하며 Python2 BINSTRING은 명시적 latin1로 bytes를 복원한다.

영수증·입력 전체 SHA/CRC/길이를 다시 대조한 뒤 제한 로더를 적용한다. 공식 pairs.txt의6000 issame와 매600개 fold의앞300 genuine/뒤300 impostor 순서를 전부 대조한다. bin 자체에 영상 이름이 없으므로 개별 인물 ID와 pairs.txt 영상 ID의 독립 대조를 완료했다고 주장하지 않는다. TopoFR 저자가 당시 평가한 파일의 전체 SHA와 동일하다는 근거도 아직 없다. 실제 primitive/영상/GPU 실행은 별도 동결·독립 검증 뒤 실행한다.
