"""최초 실행의 UTC 종료 시각을 재개에도 연장하지 않습니다."""

import json
import math
from pathlib import Path
import time

from .io import atomic_json


class DeadlineExceeded(RuntimeError):
    pass


class RunDeadline:
    def __init__(self, path, seconds, binding, *, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("전체 실행 시간 상한이 유효하지 않습니다")
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("binding") != binding or state.get("wall_seconds") != seconds:
                raise ValueError("최초 종료 시각의 소스·예산·자료 결합이 다릅니다")
            if not all(isinstance(state.get(key), (int, float)) and not isinstance(state[key], bool) and math.isfinite(state[key]) for key in ("started_utc", "deadline_utc")) or state["deadline_utc"] - state["started_utc"] != seconds:
                raise ValueError("저장한 실행 종료 시각이 유효하지 않습니다")
        else:
            started = float(clock())
            state = {"binding": binding, "wall_seconds": seconds, "started_utc": started, "deadline_utc": started + seconds}
            atomic_json(self.path, state)
        self.deadline_utc = state["deadline_utc"]

    def check(self):
        if self.clock() >= self.deadline_utc:
            raise DeadlineExceeded("최초 실행에 고정한 전체 UTC 종료 시각에 도달했습니다")
