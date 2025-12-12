import json, os, sys, time
from typing import Any, Dict, Optional


class JsonLogger:
    """Lightweight JSONL logger with severity levels."""

    def __init__(self, path: Optional[str] = None, level: str = "INFO"):
        self.path = path
        self.level = level
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Touch the file so tail -f works immediately
            with open(path, "a", encoding="utf-8"):
                pass

    def _write(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            sys.stdout.write(line + "\n")

    def log(self, level: str, msg: str, **fields: Any) -> None:
        if _level_value(level) < _level_value(self.level):
            return
        rec = {"ts": time.time(), "level": level.upper(), "msg": msg}
        if fields:
            rec.update(fields)
        self._write(rec)

    def info(self, msg: str, **fields: Any) -> None:
        self.log("INFO", msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self.log("WARNING", msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self.log("ERROR", msg, **fields)


def _level_value(level: str) -> int:
    order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    return order.get(level.upper(), 20)
