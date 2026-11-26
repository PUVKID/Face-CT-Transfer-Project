"""기존 읽기 전용 서버를 별도 포트에서 실행합니다."""
import argparse
import os
from pathlib import Path
from server import create_server

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8892)
    args = parser.parse_args()
    state_path = os.environ.get('FACE_CT_STATE_DIR')
    if not state_path:
        parser.error('FACE_CT_STATE_DIR에 기존 상태 폴더를 지정해 주세요')
    state_dir = Path(state_path).expanduser().resolve()
    if not state_dir.is_dir():
        parser.error('FACE_CT_STATE_DIR이 존재하는 폴더가 아닙니다')
    dist = ROOT / 'dashboard' / 'frontend' / 'dist'
    if not (dist / 'index.html').is_file():
        parser.error('화면 빌드 파일을 찾을 수 없습니다')
    with create_server(state_dir, dist, args.port) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
