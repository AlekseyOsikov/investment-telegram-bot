"""Аудит-лог: каждый запрос/ответ — одной JSONL-строкой.

Пишем только замаскированные данные: сырые секреты в лог не попадают ни в
``request_preview``, ни в ``input_findings``. Файл дописывается, поэтому его
можно смотреть через ``jq``/``grep`` и он переживает рестарт процесса.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class AuditLogger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: dict) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
