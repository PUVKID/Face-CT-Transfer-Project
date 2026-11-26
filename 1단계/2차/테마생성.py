"""실제 Rust 테마 컴파일러 출력을 사용자 화면에 연결합니다."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parent
FRONT=ROOT/'dashboard/frontend/src'
OFFICIAL=ROOT/'dashboard/frontend/node_modules/@astryxdesign/theme-neutral/dist/theme.css'

def main():
    executable_path=os.environ.get('ASTRYX_THEME_EXECUTABLE')
    if not executable_path:
        raise ValueError('ASTRYX_THEME_EXECUTABLE에 Rust 테마 실행 파일을 지정해 주세요')
    executable=Path(executable_path).expanduser().resolve()
    if not executable.is_file():
        raise ValueError('ASTRYX_THEME_EXECUTABLE 파일을 찾을 수 없습니다')
    if not OFFICIAL.is_file():
        raise ValueError('프런트엔드 의존성을 먼저 설치해 주세요')
    palettes=json.loads((ROOT/'테마설정.json').read_text(encoding='utf-8'))
    outputs=[]
    calls=[]
    for mode in ['light','dark']:
        argv=[str(executable),'--format','css','--mode',mode]
        for key,color in zip(['accent','background','surface','text','muted','border'],palettes[mode]):
            argv.extend(['--'+key,color])
        result=subprocess.run(argv,cwd=ROOT,capture_output=True,check=True,timeout=10)
        text=result.stdout.decode('utf-8',errors='strict')
        if '--color-primary:' not in text or '\ufffd' in text:
            raise ValueError('테마 출력 형식을 확인할 수 없습니다')
        outputs.append(text)
        calls.append({'argv':argv,'cwd':str(ROOT),'exit_code':result.returncode,'stdout_bytes':len(result.stdout),'sha256':hashlib.sha256(result.stdout).hexdigest()})
    raw='\n'.join(outputs).encode('utf-8')
    (FRONT/'astryx-theme.css').write_bytes(raw)
    (FRONT/'astryx-official.css').write_bytes(OFFICIAL.read_bytes())
    print('Rust 실제 호출2회, 라이트·다크 CSS 연결:',FRONT)

if __name__=='__main__':main()
