"""127.0.0.1 전용 읽기 API와 고정 dist 파일만 제공합니다."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark.status import read_public_status, log_tail, json_bytes
from benchmark.acquisition import atomic_json, now


def create_server(state_dir, dist, port=8765):
    state_dir, dist = Path(state_dir).resolve(), Path(dist).resolve()
    class Handler(BaseHTTPRequestHandler):
        server_version = "FaceBenchmarkLocal/1"

        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)
            category = "status" if urlparse(self.path).path == "/api/status" else "logs" if urlparse(self.path).path == "/api/logs" else "static_or_rejected"
            with (state_dir / "server.log").open("a", encoding="utf-8", newline="\n") as output:
                output.write(f"{now()} GET {category} HTTP {status}\n")

        def _json(self, status, value):
            self._send(status, json_bytes(value))

        def do_GET(self):
            host = self.headers.get("Host", "").split(":")[0]
            if host not in {"127.0.0.1", "localhost"}:
                return self._json(403, {"error": "로컬 Host만 허용합니다"})
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                if parsed.query:
                    return self._json(400, {"error": "상태 API는 입력을 받지 않습니다"})
                return self._json(200, read_public_status(state_dir))
            if parsed.path == "/api/logs":
                try:
                    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                    if not set(query).issubset({"id", "limit"}) or any(len(values) != 1 for values in query.values()):
                        raise ValueError("로그 query가 유효하지 않습니다")
                    return self._json(200, log_tail(state_dir, query.get("id", ["acquisition"])[0], int(query.get("limit", ["100"])[0])))
                except (ValueError, TypeError):
                    return self._json(400, {"error": "허용한 로그 id와 1~200 줄 수만 요청할 수 있습니다"})
            if parsed.path.startswith("/api"):
                return self._json(404, {"error": "없는 읽기 API입니다"})
            decoded = unquote(parsed.path)
            parts = Path(decoded.lstrip("/")).parts
            if ".." in parts or "\\" in decoded or any(part.startswith(".") for part in parts):
                return self._json(403, {"error": "정적 경로를 거부했습니다"})
            target = dist / (decoded.lstrip("/") or "index.html")
            if not target.resolve().is_relative_to(dist) or target.is_symlink():
                return self._json(403, {"error": "정적 root 밖의 파일입니다"})
            if not target.is_file():
                return self._json(404, {"error": "화면 빌드 파일이 없습니다"})
            if target.stat().st_size > 10 * 1024 ** 2:
                return self._json(413, {"error": "정적 출력 상한을 넘었습니다"})
            return self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

        def _reject(self):
            self._json(405, {"error": "읽기 GET만 지원합니다"})

        do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _reject

        def log_message(self, *args):
            pass
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    work = ROOT.parent / "work/stage1_revision2"
    work.mkdir(parents=True, exist_ok=True)
    server = create_server(work, ROOT / "dashboard/frontend/dist", args.port)
    atomic_json(work / "server_process.json", {"pid": os.getpid(), "started_at": now(), "status": "running", "host": "127.0.0.1", "port": server.server_port, "url": f"http://127.0.0.1:{server.server_port}", "argv": sys.argv, "cwd": os.getcwd()})
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
